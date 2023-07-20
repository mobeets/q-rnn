#%% imports

import os.path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
from analysis.decoding_beron import add_history_cols, pull_sample_dataset, fit_logreg_policy, compute_logreg_probs

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.figsize'] = [3.0, 3.0]
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

#%% load data

data = pd.read_csv(os.path.join('data/mouse/mouse_data.csv'))
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

#%% decoding to predict next action

pm1 = lambda x: 2 * x - 1
feature_functions = [
    lambda cs, rs: pm1(cs),                # choices
    lambda cs, rs: rs,                     # rewards
    lambda cs, rs: pm1(cs) * rs,           # +1 if choice=1 and reward, 0 if no reward, -1 if choice=0 and reward
    lambda cs, rs: -pm1(cs) * (1-rs),       # -1 if choice=1 and no reward, 1 if reward, +1 if choice=0 and no reward
    lambda cs, rs: (cs == rs),
    lambda cs, rs: (cs != rs),
    lambda cs, rs: pm1((cs == rs)),
    lambda cs, rs: np.ones(len(cs))        # overall bias term
]

feature_params = {
    'A': 5, # choice history
    'R': 0, # reward history
    'A*R': 5, # choice * reward history (original)
    'A*(R-1)': 5, # -choice * reward history
    'A==R': 0, # choice == reward history
    'A!=R': 0, # choice != reward history
    'B': 0 # belief history
}
memories = [y for x,y in feature_params.items()] + [1]
names = ['{}(t-{})'.format(name, t) for name, ts in feature_params.items() for t in range(ts)]

lr = fit_logreg_policy(train_features, memories, feature_functions) # refit model with reduced histories, training set
model_probs, lls, std_errors = compute_logreg_probs(test_features, [lr, memories], feature_functions)

plt.figure(figsize=(3,2))
plt.plot(lr.coef_[0,:-1], '.')
for i, (w, se) in enumerate(zip(lr.coef_[0,:-1], std_errors[:-1])):
    plt.plot([i,i], [w-se, w+se], 'k-', alpha=1.0, linewidth=1, zorder=-1)
plt.plot(plt.xlim(), [0, 0], 'k-', alpha=0.3, linewidth=1, zorder=-2)
plt.xticks(ticks=range(len(names)), labels=names, rotation=90)
plt.ylabel('weight')
plt.title('LL={:0.3f}'.format(np.mean(lls)))
plt.show()
print('ll: {:0.3f}'.format(np.mean(lls)))
