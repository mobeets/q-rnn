import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from tasks.beron2022 import get_action

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

def encode_session(choices, rewards, featparams, featfun, normalize=False):
    # Construct the features
    features = []
    for key, memory in featparams.items():
        if key not in featfun:
            raise Exception("Unrecognized feature function name: {}".format(key))
        fn = featfun[key]
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
    lr, featparams = lr_args
    model_probs = []
    lls = []
    Xs = []
    for choices, rewards in sessions:
        X, y = encode_session(choices, rewards, featparams, featfun=featfun, normalize=normalize)
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

def fit_logreg_policy(sessions, featparams, featfun, C=1.0, normalize=False):
    encoded_sessions = [encode_session(*session, featparams, featfun=featfun, normalize=normalize) for session in sessions]
    X = np.row_stack([session[0] for session in encoded_sessions])
    y = np.concatenate([session[1] for session in encoded_sessions])
    
    # Construct the logistic regression model and fit to training sessions
    lr = LogisticRegression(C=C, fit_intercept=False)
    lr.fit(X, y)
    return lr

#%% RNN/MOUSE SPECIFIC

pm1 = lambda x: 2 * x - 1

feature_functions = {
    'choice': lambda cs, rs: pm1(cs),              # choices
    'reward': lambda cs, rs: rs,                   # rewards
    'choice*reward': lambda cs, rs: pm1(cs) * rs,          # +1 if A, -1 if B; 0 if R==0
    '-choice*omission': lambda cs, rs: -pm1(cs) * (1-rs),  # +1 if b, -1 if a; 0 if R==1
    'A': lambda cs, rs: (pm1(cs) == 1) * (pm1(rs) == 1),   # A
    'a': lambda cs, rs: (pm1(cs) == 1) * (pm1(rs) == -1),  # a
    'B': lambda cs, rs: (pm1(cs) == -1) * (pm1(rs) == 1),  # B
    'b': lambda cs, rs: (pm1(cs) == -1) * (pm1(rs) == -1), # b
    'Ab': lambda cs, rs: pm1(cs == rs) * (pm1(cs == rs) > 0), # +1 if A or b; else 0
    'Ba': lambda cs, rs: pm1(cs == rs) * (pm1(cs == rs) < 0), # +1 if B or a; else 0
    'bc': lambda cs, rs: pm1(cs == rs),          # beliefs
    'bias': lambda cs, rs: np.ones(len(cs))        # overall bias term
}

def get_decoding_weights(features, feature_params):
    if 'bias' not in feature_params:
        feature_params['bias'] = 1
    assert feature_params['bias'] == 1
    names = ['{}(t-{})'.format(name, t+1) for name, ts in feature_params.items() for t in range(ts)]

    lr = fit_logreg_policy(features['train'], feature_params, feature_functions) # refit model with reduced histories, training set
    model_probs, lls, std_errors = compute_logreg_probs(features['test'], [lr, feature_params], feature_functions)
    print('ll: {:0.3f}'.format(np.mean(lls)))

    # weights = lr.coef_[0,:-1] # ignore bias term
    weights = lr.coef_[0,:]
    return weights, std_errors, names, lls

def load_mouse_data(filename='data/mouse/mouse_data.csv'):
    data = pd.read_csv(filename)
    data.head()

    probs = '80-20' # P(high)-P(low)
    seq_nback = 3 # history length for conditional probabilites
    train_prop = 0.7 # for splitting sessions into train and test
    seed = np.random.randint(1000) # set seed for reproducibility

    data = data.loc[data.Condition==probs] # segment out task condition

    data = add_history_cols(data, seq_nback) # set history labels up front

    train_session_ids, test_session_ids = train_test_split(data.Session.unique(), 
                                                        train_size=train_prop, random_state=seed) # split full df for train/test
    data['block_pos_rev'] = data['blockTrial'] - data['blockLength'] # reverse block position from transition
    data['model'] = 'mouse'
    data['highPort'] = data['Decision']==data['Target'] # boolean, chose higher probability port

    train_features, _, _ = pull_sample_dataset(train_session_ids, data)
    test_features, _, block_pos_core = pull_sample_dataset(test_session_ids, data)
    return {'train': train_features, 'test': test_features}

def get_mouse_decoding_weights(mouse_trials, feature_params):
    return get_decoding_weights(mouse_trials, feature_params)

def get_rnn_decoding_weights(AllTrials, feature_params, skip_aborts=True, skip_timeouts=True):
    rnn_features = {}
    for name in ['train', 'test']:
        rnn_features[name] = []
        for Trials in AllTrials:
            trials = Trials[name]
            # A = np.hstack([trial.A[-1] for trial in trials])
            A = np.hstack([get_action(trial, abort_value=-1 if skip_aborts else None) for trial in trials])
            R = np.hstack([trial.R[-1] for trial in trials])
            ix = np.ones(len(A)).astype(bool)
            if skip_aborts:
                ix = ix & (A >= 0)
            if skip_timeouts:
                ix = ix & (A < 2)
            A = A[ix]; R = R[ix]
            rnn_features[name].append([A,R])
    return get_decoding_weights(rnn_features, feature_params)
