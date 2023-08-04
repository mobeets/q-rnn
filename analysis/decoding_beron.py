import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def list_to_str(seq):
    seq = [str(el) for el in seq] # convert element of sequence to string
    return ''.join(seq) # flatten list to single string

def encode_as_ab(row, symm):
    if int(row.decision_seq[0]) & symm: # symmetrical mapping based on first choice in sequence 1 --> A
        mapping = {('0','0'): 'b', ('0','1'): 'B', ('1','0'): 'a', ('1','1'): 'A'} 
    elif (int(row.decision_seq[0])==0) & symm: # symmetrical mapping for first choice 0 --> A    
        mapping = {('0','0'): 'a', ('0','1'): 'A', ('1','0'): 'b', ('1','1'): 'B'} 
    else: # raw right/left mapping (not symmetrical)
        mapping = {('0','0'): 'r', ('0','1'): 'R', ('1','0'): 'l', ('1','1'): 'L'}
    return ''.join([mapping[(c,r)] for c,r in zip(row.decision_seq, row.reward_seq)])

def add_history_cols(df, N):
    from numpy.lib.stride_tricks import sliding_window_view
    
    df['decision_seq']=np.nan # initialize column for decision history (current trial excluded)
    df['reward_seq']=np.nan # initialize column for laser stim history (current trial excluded)

    df = df.reset_index(drop=True) # need unique row indices (likely no change)

    for session in df.Session.unique(): # go by session to keep boundaries clean

        d = df.loc[df.Session == session] # temporary subset of dataset for session
        df.loc[d.index.values[N:], 'decision_seq'] = \
                                    list(map(list_to_str, sliding_window_view(d.Decision.astype('int'), N)))[:-1]

        df.loc[d.index.values[N:], 'reward_seq'] = \
                                    list(map(list_to_str, sliding_window_view(d.Reward.astype('int'), N)))[:-1]

        df.loc[d.index.values[N:], 'history'] = \
                                    df.loc[d.index.values[N:]].apply(encode_as_ab, args=([True]), axis=1)

        df.loc[d.index.values[N:], 'RL_history'] = \
                                    df.loc[d.index.values[N:]].apply(encode_as_ab, args=([False]), axis=1)
        
    return df

def pull_sample_dataset(session_id_list, data):
    # choices and rewards (second idx, {0,1}) by session (first idx {0:nSessions}) for models
    sample_features = [[data[data.Session==session].Decision.values.astype('int'), \
                        data[data.Session==session].Reward.values.astype('int')] for session in session_id_list]
    sample_target = [data[data.Session==session].Target.values.astype('int') for session in session_id_list] # for expected reward only

    # makde test_df ordered same as test_sessions
    sample_block_pos_core = pd.concat([data[data.Session == session] for session in session_id_list] ).reset_index(drop=True)
    
    return sample_features, sample_target, sample_block_pos_core

def encode_session(choices, rewards, memories, featfun, normalize=False):
    assert len(memories) == len(featfun)
    # Construct the features
    features = []
    for fn, memory in zip(featfun, memories): 
        for lag in range(1, memory+1):
            # encode the data and pad with zeros
            x = fn(choices[:-lag], rewards[:-lag])
            x = np.concatenate((np.zeros(lag), x))
            features.append(x)
    features = np.column_stack(features)
    if normalize:
        features = preprocessing.StandardScaler().fit(features).transform(features)
    return features, choices

def compute_logreg_probs(sessions, lr_args, featfun, normalize=False):
    lr, memories = lr_args
    model_probs = []
    lls = []
    Xs = []
    for choices, rewards in sessions:
        X, y = encode_session(choices, rewards, memories, featfun=featfun, normalize=normalize)
        policy = lr.predict_proba(X)#[:, 1]
        ll = -log_loss(y, policy)
        model_probs.append(policy)
        lls.append(ll)
        Xs.append(X)

    X_mat = np.vstack(Xs)
    p_one_minus_p_probs = np.product(np.vstack(model_probs), axis=1).reshape(-1,1)
    cov_mat = np.linalg.pinv(np.dot((X_mat * p_one_minus_p_probs).T, X_mat))
    std_errors = np.sqrt(np.diag(cov_mat))
    return model_probs, lls, std_errors

def fit_logreg_policy(sessions, memories, featfun, C=1.0, normalize=False):
    encoded_sessions = [encode_session(*session, memories, featfun=featfun, normalize=normalize) for session in sessions]
    X = np.row_stack([session[0] for session in encoded_sessions])
    y = np.concatenate([session[1] for session in encoded_sessions])
    
    # Construct the logistic regression model and fit to training sessions
    lr = LogisticRegression(C=C, fit_intercept=False)
    lr.fit(X, y)
    return lr
