#%% imports

import os.path
import glob
import json
import numpy as np
import torch
from model import DRQN
from tasks.beron2022 import Beron2022, Beron2022_TrialLevel, BeronCensorWrapper, BeronWrapper, get_action
from tasks.wrappers import PreviousRewardWrapper, PreviousActionWrapper
from train import probe_model
from analyze import add_beliefs_beron2022
from analysis.correlations import analyze, rsq
device = torch.device('cpu')

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.figsize'] = [3.0, 3.0]
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

#%% plot model scores, before/after training

kwd = 'lowgamma'
# kwd = '_ts3'
# kwd = 'tspen_1969'
# kwd = '_ts3'
kwd = 'softmaxtest'
# kwd = 'kltest6'
# kwd = 'tspen4v2'
# kwd = 'tspen4v4'
# kwd = '_tspen4_'
# kwd = '_tspen5_'
fnms = glob.glob(os.path.join('data', 'models', '*{}*.json'.format(kwd)))

print('Found {} models.'.format(len(fnms)))
keepers = []
for i, fnm in enumerate(fnms):
    res = json.load(open(fnm))
    scs = res['scores']
    if scs[-1] > scs[0]+0.01:
        keepers.append(fnm)
    plt.plot(i, scs[0], 'ko', alpha=0.3)
    plt.plot(i, scs[-1], 'r.', alpha=0.3)
    plt.plot(i, max(scs), 'r*', alpha=0.3)
print('...and {} keepers.'.format(len(keepers)))

#%% eval many models

import glob
import os.path
from session import eval_model
from plotting.behavior import plot_example_actions, plot_average_actions_around_switch, plot_switching_by_symbol, plot_decoding_weights_grouped
from analysis.decoding_beron import get_rnn_decoding_weights, get_mouse_decoding_weights, load_mouse_data
from plotting.behavior import plot_decoding_weights, mouseWordOrder

# epsilon = 0.001; tau = None
epsilon = None; tau = 0.001
# ntrials = 10000
ntrials = 1000

# 'grant': H=10 trial-level, 'granz': timestep; 'grans': H=2 timestep; 'granasoft': H=10 trial-level w/ softmax; 'granb': H=3 trial-level; 'lowgamma': H=10 trial-level, γ=0.2
# 'tspen_1969': min_iti:2, max_iti:7, reward_delay:1, abort_penalty:0
# 'tspen2': min_iti:1, max_iti:2, reward_delay:0, abort_penalty:-0.1
# note: grans/granz are not trained well, so they're basically useless
# note: for trial-level models, default was γ=0.9. does that affect the model results?
# kwd = 'tspen4v3'
kwd = 'tspen4v4'
# kwd = 'tspen4v5'
# kwd = 'tspen4v6'
kwd = '_tspen4_'
# kwd = 'tskl3'
# kwd = '_tspen5_'
kwd = 'softmax5'
fnms = glob.glob(os.path.join('data', 'models', '*{}*.json'.format(kwd)))
# fnms = keepers
# fnms = fnms[2:3]
print('Found {} models.'.format(len(fnms)))

AllTrials = []
AllTrialsRand = []
perfs = []
for fnm in fnms:
    args = json.load(open(fnm))
    args['kl_penalty'] = 1; args['margpol_alpha'] = 0.99

    Trials, Trials_rand, _, env = eval_model(args, ntrials, epsilon, tau)
    AllTrials.append(Trials)
    AllTrialsRand.append(Trials_rand)

    # get reward rate as mean number of correct trials
    rr = np.mean([trial.R.sum()>0 for trial in Trials['train']])
    abort_rate = np.mean([trial.R.min()<0 for trial in Trials['train']])
    decs = np.array([get_action(trial, abort_value=-1) for trial in Trials['train']])
    margpol = {d: np.round(np.mean(decs == d),2) for d in np.unique(decs)}
    print(f'correct rate: {rr:0.2f}, abort rate: {abort_rate:0.2f}, marg pol: {margpol}')
    perfs.append((rr, abort_rate))

if len(perfs) > 1:
    perfs = np.vstack(perfs)
    plt.bar(np.arange(len(perfs)), perfs.sum(axis=1))
    plt.bar(np.arange(len(perfs)), perfs[:,0])
    plt.xlabel('model index'), plt.ylabel('outcomes'), plt.ylim([0,1]), plt.show()

#%% visualize policy before/after KL term is added to policy

import torch.nn as nn

showPrefs = True

Q = np.vstack([trial.Q for trial in Trials['train']])
A = np.hstack([trial.A for trial in Trials['train']])
H = Q/tau
Pol = nn.functional.softmax(torch.Tensor(H), dim=-1).numpy()

t1 = 500
t2 = t1 + 25

kl_alpha = 0.99999
kl_beta = 150

plt.figure(figsize=(6,3))

margpol = np.ones(3)/3
MargPol = []
for pol in Pol:
    margpol = (1-kl_alpha)*margpol + kl_alpha*pol
    MargPol.append(margpol)
MargPol = np.vstack(MargPol)
# MargPol = M4['P'] # todo: compute marginal policy given kl_alpha

H_pc = H + kl_beta*MargPol
Pol_pc = nn.functional.softmax(torch.Tensor(H_pc), dim=-1).numpy()

if showPrefs:
    plt.plot(H)
    plt.gca().set_prop_cycle(None)
    plt.plot(H_pc, '--')
else:
    plt.plot(Pol)
    plt.gca().set_prop_cycle(None)
    plt.plot(Pol_pc - 1.2)

plt.xlabel('time')
plt.ylabel('action prefs' if showPrefs else 'π(a | s)')
plt.xlim([t1, t2])

#%% plot behavior as a function of observation history

wordSize = 2
plot_switching_by_symbol([Trials['test'] for Trials in AllTrials], wordOrder=None, wordSize=wordSize)
plot_switching_by_symbol([Trials['test'] for Trials in AllTrials], wordOrder=mouseWordOrder[wordSize], wordSize=wordSize)
plot_average_actions_around_switch([Trials['test'] for Trials in AllTrials])

#%% plot decoding weights

showMouse = False
feature_params = {
    'choice': 2, # choice history
    'reward': 2, # reward history
    'choice*reward': 5, # rewarded trials, aligned to action (original)
    '-choice*omission': 5, # unrewarded trials, aligned to action
    'A': 0, # rewarded trials, left choice
    'a': 0, # unrewarded trials, left choice
    'B': 0, # rewarded trials, right choice
    'b': 0, # unrewarded trials, right choice
    'Ab': 0,
    'Ba': 0,
    'bc': 0 # belief history
}

weights, std_errors, names, lls = get_rnn_decoding_weights(AllTrials, feature_params)
plot_decoding_weights_grouped(weights, std_errors, feature_params, title='Value RNN')

if showMouse:
    if 'mouse_trials' not in vars():
        mouse_trials = load_mouse_data()
    weights, std_errors, names, lls = get_mouse_decoding_weights(mouse_trials, feature_params)
    plot_decoding_weights_grouped(weights, std_errors, feature_params, title='Mouse')

#%% visualize RPEs (timestep-level only)

from plotting.behavior import get_action

scale = 1 # scale=100 in old models where we had r/100
gamma = 0.9
trials = AllTrials[0]['test']#[:50]
V = np.hstack([[q[a] for a,q in zip(trial.A, trial.Q)] for trial in trials])
Qmax = np.hstack([[q.max() for q in trial.Q] for trial in trials])
R = np.hstack([trial.R for trial in trials])
RPE = R[:-1]/scale + gamma*Qmax[1:] - V[:-1]
RPE = np.hstack([np.nan, RPE[:-1]]) # delay by one since RPE[t] requires next obs

min_iti = np.min([trial.iti for trial in trials])
max_iti = np.max([trial.iti for trial in trials])
max_n = max_iti + np.max([len(trial.X[trial.iti:]) for trial in trials]) + min_iti # max trial length

i = 0
for trial in trials:
    trial.V = V[i:(i+len(trial)+min_iti)]
    trial.RPE = RPE[i:(i+len(trial)+min_iti)]
    i += len(trial)

ys = {}
conds = []
for i,trial in enumerate(trials[:-1]): # ignore last trial because RPE is shorter
    if trial.R.sum() < 0:
        cond = 'aborted'
    elif trial.R.sum() == 1:
        cond = 'rewarded'
    elif get_action(trial) == 2:
        cond = 'timeout'
    else:
        cond = 'unrewarded'
    conds.append(cond)
    if cond == 'aborted':
        continue
    t_start = max([0, trial.iti-max_iti])
    yv = trial.V[t_start:]
    yr = trial.RPE[t_start:]
    yr[0] = np.nan # ignore RPE from previous trial

    # get recent reward count
    if i > 3:
        # recent rewards in last 3 trials, each either -1 or 1
        rs = 2*np.array([trial.R.sum() for trial in trials[(i-3):i]])-1
        # count recent rewards how I think celia does
        if len(np.unique(rs)) == 1:
            nrs = rs.sum()
        elif rs[-1] != rs[-2]:
            nrs = rs[-1]
        elif rs[-1] != rs[-3]:
            nrs = rs[-2:].sum()
        else:
            assert False
    else:
        nrs = np.nan
    nrs = nrs*np.ones(len(yv))

    if trial.iti < max_iti:
        nc = max_iti - trial.iti
        yv = np.hstack([[np.nan]*nc, yv])
        yr = np.hstack([[np.nan]*nc, yr])
        nrs = np.hstack([[np.nan]*nc, nrs])
    if len(yv) < max_n:
        yv = np.hstack([yv, [np.nan]*(max_n-len(yv))])
        yr = np.hstack([yr, [np.nan]*(max_n-len(yr))])
        nrs = np.hstack([nrs, [np.nan]*(max_n-len(nrs))])

    if cond not in ys:
        ys[cond] = []
    ys[cond].append((yv, yr, nrs))

for cond in ys:
    ys[cond] = np.dstack(ys[cond])

ncols = 1; nrows = 2
# ncols = 2; nrows = 1
clrs = {'rewarded': np.array([45, 107, 207])/255, 'unrewarded': [0.8,0.2,0.2]}
plt.figure(figsize=(3*ncols,3*nrows))
for i in range(2):
    plt.subplot(nrows,ncols,i+1)
    for cond, vs in ys.items():
        if cond not in ['rewarded', 'unrewarded']:
            continue
        vsc = vs[i,:,:]
        if len(vsc) == 0:
            continue
        mu = np.nanmean(vsc, axis=1)
        xs = np.arange(len(mu)) - max_iti - 1
        plt.plot(xs, mu, '.-', label=cond, color=clrs.get(cond, 'k'))
    plt.title('Value' if i==0 else 'RPE')
    plt.xlabel('time rel. to go cue')
    plt.ylabel('Value' if i==0 else 'RPE')
    plt.legend(fontsize=10, loc='lower left')
    plt.plot(plt.xlim(), [0,0], 'k-', zorder=-1, alpha=0.5)
plt.tight_layout()
plt.show()

clrs = plt.get_cmap('RdBu', 6)
plt.figure(figsize=(3*ncols,3*nrows))
for i in range(2):
    plt.subplot(nrows,ncols,i+1)
    for cond, vs in ys.items():
        if cond not in ['rewarded', 'unrewarded']:
            continue
        nrs = vs[-1,:,:]
        vsc = vs[i,:,:]
        if len(vsc) == 0:
            continue
        for nr in [-3,-2,-1,1,2,3]:
            ix = np.nanmax(nrs, axis=0) == nr
            if ix.sum() == 0:
                continue
            mu = np.nanmean(vsc[:,ix], axis=1)
            xs = np.arange(len(mu)) - max_iti - 1
            plt.plot(xs, mu, '.-' if cond == 'rewarded' else '.--',
                     label=nr, color=clrs(nr+3))
    # plt.xlim([-0.5,0.5])
    plt.title('Value' if i==0 else 'RPE')
    plt.xlabel('time rel. to go cue')
    plt.ylabel('Value' if i==0 else 'RPE')
    # plt.legend(fontsize=10, loc='lower left')
    plt.plot(plt.xlim(), [0,0], 'k-', zorder=-1, alpha=0.5)
plt.tight_layout()

#%% visualize trial data

t = 0

trial = trials[t]
V = np.array([q[a] for q,a in zip(trial.Q, trial.A)])
Qmax = np.array([q.max() for q in trial.Q])
R = trial.R
RPE = R[:-1] + gamma*Qmax[1:] - V[:-1]

plt.plot(np.arange(len(trials[t].V))-trials[t].iti-1, trials[t].V, '.-')
# plt.plot(np.arange(len(RPE))-trials[t].iti-1, RPE, '.-')
# plt.plot(np.arange(len(trials[t].V))-trials[t].iti-1, trials[t].RPE, '.-')

print(np.round(np.hstack([trials[t].X, trials[t].A[:,None], trials[t].R[:,None], 10*trials[t].V[:len(trials[t]),None],  10*trials[t].Q]),1))

t += 1
print(np.round(np.hstack([trials[t].X, trials[t].A[:,None], trials[t].R[:,None], 10*trials[t].V[:len(trials[t]),None],  10*trials[t].Q]),1))

#%%

eps_start = 0.6 # initial epsilon used in policy
eps_end = 0.001 # final epsilon used in policy
eps_decay = 0.995 # time constant of decay for epsilon used in policy
epsilon = eps_start
es = []
for _ in range(300):
    es.append(epsilon)
    epsilon = max(eps_end, epsilon * eps_decay)

plt.plot(es)

#%% plot belief R^2

Rsqs = []
for Trials, Trials_rand in zip(AllTrials, AllTrialsRand):
    results_rand = analyze(Trials_rand, key='Z', onlyLastTimestep=True)
    results = analyze(Trials, key='Z', onlyLastTimestep=True)
    Rsqs.append((results_rand['rsq'], results['rsq']))
Rsqs = np.vstack(Rsqs)

mus = np.mean(Rsqs, axis=0)
ses = np.std(Rsqs, axis=0)/np.sqrt(len(Rsqs))
names = ['Untrained\nRQN', 'RQN']

plt.figure(figsize=(1.4,2))
for i,(mu,se,name) in enumerate(zip(mus,ses,names)):
    plt.plot(i, mu, 'ko', zorder=1, alpha=0.8)
    plt.plot(i*np.ones(2), [mu-se, mu+se], 'k-', zorder=-1)
    ys = Rsqs[:,i]
    xs = i*np.ones(len(ys))
    xs += 0.2*(np.random.rand(len(xs))-0.5)
    plt.plot(xs, ys, '.', markersize=8, markerfacecolor='lightgray',
             alpha=1, zorder=-2, markeredgecolor='darkgray')
plt.xlim([-0.5, len(mus)-1+0.5])
plt.ylim([-0.1, 1.1])
plt.xticks(ticks=[0,1], labels=names, rotation=90)
plt.ylabel('Belief $R^2$')
plt.tight_layout()

#%% example belief agent

from tasks.beron2022 import BeronBeliefAgent

ntrials = 5000
env_params = {'p_rew_max': 0.8, 'p_switch': 0.02, 'ntrials': ntrials}
env = Beron2022_TrialLevel(**env_params)
env = PreviousRewardWrapper(env)
env = PreviousActionWrapper(env, env.action_space.n)
agent = BeronBeliefAgent(env)

epsilon = 0.04

nreps = 10
BeliefTrials = []
seeds = [456, 787]
seeds = [None, None]
for i in range(nreps):
    Trials = {}
    for name, seed in zip(['train', 'test'], seeds):
        # reset seeds
        env.state = None
        if seed is not None:
            env.reset(seed=seed)
            agent.reset(seed=seed+1)
        trials = probe_model(agent, env, behavior_policy=None, epsilon=epsilon, tau=None, nepisodes=1)
        Trials[name] = trials
    BeliefTrials.append(Trials)

# analyze belief agent
from plotting.behavior import mouseWordOrder
plot_switching_by_symbol([Trials['test'] for Trials in BeliefTrials], wordOrder=None)
plot_switching_by_symbol([Trials['test'] for Trials in BeliefTrials], wordOrder=mouseWordOrder)
plot_example_actions(Trials['test'])
plot_average_actions_around_switch([Trials['test'] for Trials in BeliefTrials])

#%%

feature_params = {
    'choice': 1, # choice history
    'reward': 1, # reward history
    'choice*reward': 5, # rewarded trials, aligned to action (original)
    '-choice*omission': 5, # unrewarded trials, aligned to action
    'A': 0, # rewarded trials, left choice
    'a': 0, # unrewarded trials, left choice
    'B': 0, # rewarded trials, right choice
    'b': 0, # unrewarded trials, right choice
    'Ab': 0,
    'Ba': 0,
    'bc': 0 # belief history
}
weights, std_errors, names, lls = get_rnn_decoding_weights(BeliefTrials, feature_params)
plot_decoding_weights_grouped(weights, std_errors, feature_params, title='Beliefs')

#%% visualize beliefs

from tasks.beron2022 import belief_step_beron2022
b_prev = 0.9
rows = []
for r_prev in [0,1]:
    for a_prev in [0,1]:
        b_next = belief_step_beron2022(b_prev, r_prev, a_prev, env.p_rew_max, env.p_switch)
        rows.append((r_prev, a_prev, b_next))
rows = np.vstack(rows)
print(rows)

#%% visualize beliefs

def get_name(a, r, collapse=False):
    if (a == r) and (a == 1):
        name = 'Stay' if collapse else 'A'
    elif (a != r) and (a == 1):
        name = 'Switch' if collapse else 'a'
    elif (a == r) and (a == 0):
        name = 'Stay' if collapse else 'b'
    elif (a != r) and (a == 0):
        name = 'Switch' if collapse else 'B'
    else:
        assert False
    return name

from tasks.beron2022 import belief_step_beron2022
data = {key: [] for key in rows}

b_prevs = np.linspace(0,1,11)
for b_prev in b_prevs:
    rows = {}
    for r_t2 in [0,1]:
        for a_t2 in [0,1]:
            b_t2 = belief_step_beron2022(b_prev, r_t2, a_t2, env.p_rew_max, env.p_switch)
            for r_t1 in [0,1]:
                for a_t1 in [0,1]:
                    b_t1 = belief_step_beron2022(b_t2, r_t1, a_t1, env.p_rew_max, env.p_switch)
                    key = get_name(a_t2, r_t2) + '-' + get_name(a_t1, r_t1)
                    if key not in rows:
                        rows[key] = []
                    val = (b_t2, b_t2-b_prev, b_t1, b_t1-b_t2, b_t1-b_prev)
                    rows[key].append(val)

    for key, vals in rows.items():
        assert len(np.unique(vals, axis=0)) == 1
        # print(key, np.round(vals[0], 3))
        data[key].append(vals[0][-1])

X,y  = [],[]
for key in data:
    parts = key.split('-')
    vals = data[key]

    xs = np.zeros(8)
    if parts[1] == 'A':
        xs[0] = 1
    elif parts[1] == 'b':
        xs[1] = 1
    elif parts[1] == 'B':
        xs[2] = 1
    else:
        xs[3] = 1
    if parts[0] == 'A':
        xs[4] = 1
    elif parts[0] == 'b':
        xs[5] = 1
    elif parts[0] == 'B':
        xs[6] = 1
    else:
        xs[7] = 1
    
    # xs = np.zeros(4)
    # if parts[1] == 'Switch':
    #     xs[0] = 1
    # else:
    #     xs[2] = 1
    # if parts[0] == 'Switch':
    #     xs[1] = 1
    # else:
    #     xs[3] = 1
    
    Xc = np.tile(xs[None,:], (len(vals), 1))
    X.append(Xc)
    y.append(vals)

X = np.vstack(X)
y = np.hstack(y)[:,None]

from analysis.correlations import linreg_fit

res = linreg_fit(X, y, scale=False, add_bias=True)
ws = res['W'][:-1]
ws = np.reshape(ws, (2,4)).T
plt.plot(ws.T, '.-', alpha=0.5)

# plt.plot(b_prevs, np.vstack([data[x] for x in data]).T)
# plt.plot(np.vstack([data[x] for x in data]).mean(axis=1))

#%% fit belief regression model

def get_name(a, r, collapse=False):
    if (a == r) and (a == 1):
        name = 'Stay' if collapse else 'A'
    elif (a != r) and (a == 1):
        name = 'Switch' if collapse else 'a'
    elif (a == r) and (a == 0):
        name = 'Stay' if collapse else 'b'
    elif (a != r) and (a == 0):
        name = 'Switch' if collapse else 'B'
    else:
        assert False
    return name

env_params = {'p_rew_max': 0.8, 'p_switch': 0.02, 'ntrials': ntrials}
env = Beron2022_TrialLevel(**env_params)
env = PreviousRewardWrapper(env)
env = PreviousActionWrapper(env, env.action_space.n)
obs, info = env.reset()
done = False

b = 0.5
X, y = [], []
S = [info['state']]
while not done:
    a = np.random.choice(2)
    obs, r, done, _, info = env.step(a)
    b = belief_step_beron2022(b, r, a, env.p_rew_max, env.p_switch)

    name = get_name(a, r)
    x = np.array([name==nm for nm in ['A', 'a', 'b', 'B']]).astype(float)
    assert x.sum() == 1
    X.append(x)
    y.append(b)
    S.append(info['state'])

X = np.vstack(X)
y = np.hstack(y)[:,None] - 0.5
S = np.hstack(S)

plt.plot(S), plt.xlim([0,200])

from analysis.correlations import linreg_fit, linreg_eval

mdl = linreg_fit(X, y, scale=False, add_bias=False)
res = linreg_eval(X, y, mdl)
ws = mdl['W']#[:-1]
# plt.plot(ws, '.-', alpha=0.5)
print(res['rsq'])

#%%

env_params = {'p_rew_max': 0.8, 'p_switch': 0.02, 'ntrials': 200, 'reward_delay': 1}
# env_params = {'p_rew_max': 0.8, 'p_switch': 0.02, 'ntrials': 200, 'iti_min': 1, 'iti_max': 6, 'iti_dist': 'uniform', 'reward_delay': 1}
env = Beron2022(**env_params)
env = PreviousRewardWrapper(env)
env = PreviousActionWrapper(env, env.action_space.n)
obs, info = env.reset()
done = False

S = [list(info.values())]
X = [obs]
while not done:
    a = np.random.choice(2)
    obs, r, done, _, info = env.step(a)
    S.append(list(info.values()))
    X.append(obs)

S = np.vstack(S)
X = np.vstack(X)
# plt.plot(X), plt.xlim([0,200])

print(len(X))

#%%

ts = np.arange(20)
itis = np.arange(4)
vals = []
for iti in itis:
    val = 0.9**(iti + 15-ts)
    val[val > 1] = 0
    vals.append(val)
    # plt.plot(val)

vals = np.vstack(vals).T
for i,iti in enumerate(itis):
    vals[(iti+1):,i] = np.nan
# plt.plot(vals, '.-')
    
V = np.nanmean(vals, axis=1)
plt.plot(V, '.-')
plt.plot(0 - 0.9*V, '.-')

#%% load model

# run_name = 'h3_beron_v3_p08'
# run_name = 'h3_beron_v5_p08_softmax'
# run_name = 'h3_beron_v5_p08_epsilon'
# run_name = 'h3_beron_v6_p08_epsilon'
# run_name = 'h2_beron_v7_p08_epsilon'
# run_name = 'h6_beron_v8_p08_epsilon'
# run_name = 'h6_beron_v9_p08_epsilon'
# run_name = 'h2_beron_v10_p08_epsilon'
# run_name = 'h2_beron_v11_p08_epsilon'
# run_name = 'h2_beron_v12_p08_epsilon'
# run_name = 'h3_beron_v13_p1_epsilon'
# run_name = 'h10_2023-07-21-11-19-17-039345'
# run_name = 'h10_2023-07-21-11-19-17-039345'
run_name = 'h10_2023-07-21-13-20-20-175694'
# run_name = 'h2_2023-07-21-13-30-18-495491'
run_name = 'h2_2023-07-21-14-21-40-357992'
run_name = 'h2_2023-07-21-14-33-29-841107'
run_name = 'h2_2023-07-21-14-57-55-870084' # random policy all thru training
run_name = 'h3_beronwrap'
run_name = 'h2_beronwrap'
run_name = 'h2_beronwrap_rand'
run_name = 'h2_beronwraprnn'
run_name = 'h2_beronnogamma'
run_name = 'h2_beronp1'
run_name = 'h2_beronp1v2'
run_name = 'h2_beronp1v3'
run_name = 'h2_beronnew'
run_name = 'h2_beronp1switch'
run_name = 'h3_berontime'
run_name = 'h3_berontime2'
# run_name = 'h2_berontime3'
# run_name = 'h3_berontime4'
# run_name = 'h3_berontime5'
# run_name = 'h3_berontimep1'
# run_name = 'h3_berontime6'
run_name = 'h3_berontime7'
# run_name = 'h3_brnlambda'
# run_name = 'h3_brnlambda2'
# run_name = 'h3_test'
# run_name = 'h3_test2'
# run_name = 'h3_berontime6'
# run_name = 'h3_test4a'
run_name = 'h3_proto'
run_name = 'h5_proto2'
run_name = 'h10_proto3'
run_name = 'h10_grant1'

ntrials = 9000
epsilon = 0.03; tau = None

args = json.load(open('data/models/results_{}.json'.format(run_name)))
env_params = {
    'p_rew_max': args.get('p_reward_max', 0.8),
    'p_switch': args.get('p_switch', 0.02),
    'ntrials': ntrials}
hidden_size = args['hidden_size']
modelfile = args['filenames']['weightsfile_final']
initial_modelfile = args['filenames']['weightsfile_initial']
print('H={}, prew={}, pswitch={}'.format(hidden_size, env_params['p_rew_max'], env_params['p_switch']))

if args['experiment'] == 'beron2022_time':
    env_params.update({'iti_min': args.get('iti_min', 0), 'iti_p': args.get('iti_p', 0.5), 
        'abort_penalty': args.get('abort_penalty', 0),
        'include_null_action': args.get('abort_penalty', 0) < 0})
    env = Beron2022(**env_params)
else:
    env = Beron2022_TrialLevel(**env_params)

input_size = 1 + args['include_prev_reward'] + args['include_prev_action']*env.action_space.n
if args['experiment'] == 'beron2022_time':
    input_size += args.get('include_beron_wrapper', False)

if args['include_prev_reward']:
    env = PreviousRewardWrapper(env)
if args['include_prev_action']:
    env = PreviousActionWrapper(env, env.action_space.n)
if args['include_beron_wrapper']:
    env = BeronWrapper(env, input_size)
if args.get('include_beron_censor', False):
    env = BeronCensorWrapper(env, args['include_beron_wrapper'])

model = DRQN(input_size=input_size, # empty + prev reward + prev actions
                hidden_size=hidden_size,
                output_size=env.action_space.n,
                recurrent_cell=args.get('recurrent_cell', 'gru')).to(device)
model.load_weights_from_path(modelfile)

behavior_policy = DRQN(input_size=input_size, # empty + prev reward + prev actions
                hidden_size=hidden_size,
                output_size=env.action_space.n,
                recurrent_cell=args.get('recurrent_cell', 'gru')).to(device)
behavior_policy.load_weights_from_path(initial_modelfile)
behavior_policy = None

# probe model
Trials = {}
Trials_rand = {}
for useRandomModel in [True, False]:
    if useRandomModel:
        model.reset(gain=1)
    else:
        model.load_weights_from_path(modelfile)
    
    for name, seed in {'train': 456, 'test': 787}.items():
        # reset seeds
        env.state = None
        env.reset(seed=seed)
        model.reset_rng(seed+1)
        if behavior_policy is not None:
            behavior_policy.reset_rng(seed+2)

        # run model on trials
        trials = probe_model(model, env, behavior_policy=behavior_policy,
                                epsilon=epsilon, tau=tau, nepisodes=1)
        print(useRandomModel, name, np.round(np.hstack([trial.R for trial in trials]).mean(),3))

        # add beliefs
        add_beliefs_beron2022(trials, env.p_rew_max, env.p_switch)
        ymx = np.max(np.abs(np.hstack([trial.Q[:,1]-trial.Q[:,0] for trial in trials])))
        for trial in trials:
            trial.Qdiff = (trial.Q[:,1] - trial.Q[:,0])[:,None]/ymx
        
        # discard trials at the beginning of episodes (warmup)
        trials = [trial for trial in trials if trial.index_in_episode > 5]

        if useRandomModel:
            Trials_rand[name] = trials
        else:
            Trials[name] = trials

#%% plot Fig. 1B-D from Beron et al. (2022)

from plotting.behavior import plot_example_actions, plot_average_actions_around_switch, plot_switching_by_symbol, mouseWordOrder

ntrials = 10000; epsilon = None; tau = 1#0.000001

fnms = glob.glob(os.path.join('data', 'models', '*grant*.json'))
model_file = fnms[0]
Trials, Trials_rand, model, env = eval_model(model_file, ntrials, epsilon, tau)
plot_example_actions(Trials['test'])
plot_average_actions_around_switch([Trials['test']])
plot_switching_by_symbol([Trials['test']], modelBased=True, tau=tau)#, wordOrder=mouseWordOrder)

#%%

_, switches = plot_switching_by_symbol([Trials['test']], modelBased=True, tau=0.005)#, wordOrder=mouseWordOrder)

#%% probe model on repeated sequences of inputs

tau = 0.0001
Zinit = np.vstack([trial.Z for trial in Trials['train']])
unique_obs = np.unique(np.vstack([trial.X for trial in Trials['train']]), axis=0)
nreps = 500

for j, x in enumerate(unique_obs):
    X = [x]*3
    Z = []
    Q = []
    with torch.no_grad():
        for _ in range(nreps):
            ind = np.random.choice(range(len(Zinit)))
            h = torch.tensor(Zinit[ind]).to(device).unsqueeze(0).unsqueeze(0)
            for obs in X:
                cobs = torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0)
                a, (q, h) = model.sample_action(cobs, h, epsilon=epsilon, tau=tau)
            Z.append(h.numpy().flatten())
            Q.append(q.numpy().flatten())
    Z = np.vstack(Z)
    Q = np.vstack(Q)
    Qprobs = torch.softmax(torch.Tensor(Q)/tau, axis=1).numpy()
    mu = Qprobs[:,0].mean()
    se = Qprobs[:,0].std()/np.sqrt(len(Qprobs))
    h = plt.plot(j, mu, 'o', label=x)
    plt.plot(j*np.ones(2), [mu-se, mu+se], '-', color=h[0].get_color())
plt.xticks(ticks=range(len(unique_obs)), labels=unique_obs, rotation=90)
plt.xlabel('input (repeated x3)')
plt.ylabel('$P_{\\tau}(A = 1)$')
plt.ylim([-0.03, 1.03])

#%% choice regression (using same code as for mice)

from analysis.decoding_beron import get_rnn_decoding_weights
from plotting.behavior import plot_decoding_weights

feature_params = {
    'a': 1, # choice history
    'r': 1, # reward history
    'x': 5, # rewarded trials, aligned to action (original)
    'y': 5, # unrewarded trials, aligned to action
    'b': 0 # belief history
}

weights, std_errors, names, lls = get_rnn_decoding_weights([Trials], feature_params)
plot_decoding_weights(weights, std_errors, names)

#%% accuracy

trials = Trials['test']
if args['include_beron_wrapper']:
    X = np.where(np.vstack([trial.X[0] for trial in trials]))[1][:,None]
else:
    X = np.vstack([trial.X[0] for trial in trials])
R = np.vstack([trial.R[-1] for trial in trials])[:,0]
Q = np.vstack([trial.Q[-1] for trial in trials])

for x in np.unique(X, axis=0):
    ix = np.all(X == x, axis=1)
    print('O={}, p(O)={:0.3f}, r={:0.3f}, Q={}'.format(x, np.mean(ix), R[ix].mean(), np.round(Q[ix].mean(axis=0),3)))

#%% compare beliefs and latent activity

results_rand = analyze(Trials_rand, key='Z')
print(results_rand['rsq'])
results = analyze(Trials, key='Z')
resq = analyze(Trials, key='Qdiff')
print(results['rsq'])

ys = [results_rand['rsq'], results['rsq']]#, resq['rsq']]
labels = ['Untrained\nRQN', 'RQN']#, 'RQN-ΔQ']
plt.figure(figsize=(0.5*len(ys),2))
plt.plot(range(len(ys)), ys, 'ko')
plt.xlim([-0.5, len(ys)-0.5])
plt.xticks(ticks=range(len(ys)), labels=labels, rotation=90)
plt.ylim([-0.05,1.05])
plt.ylabel('Belief $R^2$')

#%% plot belief predictions over trials

trials = Trials['test']
S = np.hstack([trial.S[-1] for trial in trials])
B = np.vstack([trial.B[-1] for trial in trials])
Bhat = np.vstack([trial.Bhat_Z[-1] for trial in trials])
Bhat_Q = np.vstack([trial.Bhat_Qdiff[-1] for trial in trials])

rsq1 = rsq(B, Bhat); rsq2 = rsq(B, Bhat_Q)
print('Bhat r^2: {:0.2f}, Qdiff r^2: {:0.2f}'.format(rsq1, rsq2))

from matplotlib.patches import Rectangle

plt.figure(figsize=(9,1.5))
xs = np.arange(len(S))

S = np.hstack([trial.S[-1] for trial in trials])
A = np.hstack([trial.A[-1] for trial in trials])
R = np.hstack([trial.R[-1] for trial in trials])

switchInds = np.hstack([0, np.where(np.diff(S) != 0)[0] + 1, len(S)+1])
for i in range(len(switchInds)-1):
    x1 = switchInds[i]
    x2 = switchInds[i+1]
    if S[x1] == 1:
        rect = Rectangle((x1-0.5, -0.05), x2-x1, 1.1, alpha=0.3)
        plt.gca().add_patch(rect)

plt.plot(xs, B, 'k-', linewidth=1, label='Beliefs')
plt.plot(xs, Bhat, 'r-', linewidth=1, alpha=0.6, label='$\widehat{B}$')
plt.plot(xs, Bhat_Q, 'b-', linewidth=1, alpha=0.6, label='$\Delta Q$')
plt.yticks(ticks=[0,1], labels=['left', 'right'])
plt.xlabel('Trial')
plt.xlim(0 + np.array([0, 140.5]))
plt.legend(fontsize=9)

#%% compare belief predictions

plt.plot(Bhat, B, 'r.', markersize=5, alpha=0.6, label='$\widehat{B}$')
plt.plot(Bhat_Q, B, 'b.', markersize=5, alpha=0.6, label='$\Delta Q$')
plt.xlabel('Belief prediction')
plt.ylabel('Belief')
plt.xticks([0,0.5,1]); plt.yticks([0,0.5,1])
plt.legend(fontsize=9)

#%% visualize within-trial activity

from analysis.pca import fit_pca, apply_pca
if '_time' not in args['experiment']:
    raise Exception("Not a relevant analysis for this type of environment.")

ninits = 100
niters = 100
showFPs = True
showTrials = True
showPCs = True
showQ = True

pca = fit_pca(Trials['train'])
trials = apply_pca(Trials['test'], pca)
if showQ:
    pca.transform = lambda z: ((z @ model.output.weight.detach().numpy().T) + model.output.bias.detach().numpy())
    lbl = 'Q'
elif not showPCs:
    pca.transform = lambda z: z
    lbl = 'Z'
else:
    lbl = 'Z'

X = np.vstack([trial.X for trial in trials])
Z = np.vstack([trial.Z for trial in trials])
Zpc = pca.transform(Z)
zmin = Z.min()-0.01
zmax = Z.max()+0.01

switchTrials = [t for t in range(len(trials)-1) if trials[t].S[0] != trials[t+1].S[0]]
switchTrial = switchTrials[1]
trials = trials[switchTrial:switchTrial+20]
alpha = 0.5

plt.figure(figsize=(3,3))
plt.plot(Zpc[:,0], Zpc[:,1], 'k.', markersize=1, alpha=0.1, zorder=-1)

if showFPs:
    obs = np.zeros(model.input_size)
    cobs = torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0)
    fps = []
    for i in range(ninits):
        zinit = np.random.rand(model.hidden_size)*(zmax-zmin) + zmin
        zinit = Z[np.random.choice(len(Z))] + 0.0*(np.random.rand(model.hidden_size)-0.5)
        # zi = pca.transform(zinit[None,:]); plt.plot(zi[:,0], zi[:,1], '.', markersize=1, alpha=0.2)

        h = torch.Tensor(zinit)[None,None,:]
        zs = []
        zs.append(h.detach().numpy().flatten())
        for _ in range(niters):
            a, (q, h) = model.sample_action(cobs, h.to(device), epsilon=0)
            zs.append(h.detach().numpy().flatten())
        zs = np.vstack(zs)
        fps.append(zs[-1])

    fps = pca.transform(np.vstack(fps))
    h = plt.plot(fps[:,0], fps[:,1], '*', color='r', markersize=5, linewidth=1, label=name, zorder=10)

if showTrials:
    for trial in trials:
        zpc = pca.transform(trial.Z)
        if trial.S[-1] == 0:
            color = 'b'
        else:
            color = 'r'
        plt.plot(zpc[0,0], zpc[0,1], '+', color=color, linewidth=1, markersize=5, alpha=alpha, zorder=0)
        plt.plot(zpc[:,0], zpc[:,1], '.-' if trial.R[-1] == 1 else '--',
                 markersize=2,
                 color=color, linewidth=1, alpha=alpha, zorder=0)
plt.tight_layout()

if showQ:
    plt.axis('equal')
    plt.plot(plt.xlim(), plt.xlim(), 'k-', alpha=0.5, linewidth=1, zorder=-1)
plt.xlabel('${}_1$'.format(lbl))
plt.ylabel('${}_2$'.format(lbl))

#%% assess RNN responses to fixed inputs

from analysis.pca import fit_pca, apply_pca
if '_time' in args['experiment']:
    raise Exception("Not a relevant analysis for this type of environment.")

ninits = 100
niters = 100
showPCs = True
showFPs = True
showQ = True

pca = fit_pca(Trials['train'])
if showQ:
    pca.transform = lambda z: (z @ model.output.weight.detach().numpy().T) + model.output.bias.detach().numpy()
    lbl = 'Q'
elif not showPCs:
    pca.transform = lambda z: z
    lbl = 'Z'
else:
    lbl = 'Z'
trials = apply_pca(Trials['test'], pca)

X = np.vstack([trial.X for trial in trials])
Z = np.vstack([trial.Z for trial in trials])
Zpc = pca.transform(Z)

zmin = Z.min()-0.01
zmax = Z.max()+0.01

plt.figure(figsize=(3,3))

for sign in [0,1,2,3]:#,4,5]:
    if args['include_beron_wrapper']:
        if sign == 0:
            obs = np.array([1,0,0,0])
            name = 'A'
        elif sign == 1:
            obs = np.array([0,1,0,0])
            name = 'b'
        elif sign == 2:
            obs = np.array([0,0,1,0])
            name = 'a'
        elif sign == 3:
            obs = np.array([0,0,0,1])
            name = 'B'
    else:
        if sign == 0:
            r_prev = 0
            a_prev = np.zeros(2); a_prev[0] = 1
        elif sign == 1:
            r_prev = 1
            a_prev = np.zeros(2); a_prev[1] = 1
        elif sign == 2:
            r_prev = 0
            a_prev = np.zeros(2); a_prev[1] = 1
        elif sign == 3:
            r_prev = 1
            a_prev = np.zeros(2); a_prev[0] = 1
        elif sign == 4:
            r_prev = 0
            a_prev = np.zeros(2)
        elif sign == 5:
            r_prev = 1
            a_prev = np.zeros(2)
        obs = np.hstack([[0], [r_prev], a_prev])
        if sign < 4:
            name = 'r={}, a={}'.format(r_prev, np.where(a_prev)[0][0])
        else:
            name = 'r={}'.format(r_prev)
    
    cobs = torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0)
    ix = np.all(X == obs, axis=1)
    hf = plt.plot(Zpc[ix,0], Zpc[ix,1], '.', markersize=1, alpha=0.3, zorder=-1, label=name if not showFPs else '_')

    if showFPs:
        fps = []
        for i in range(ninits):
            zinit = np.random.rand(model.hidden_size)*(zmax-zmin) + zmin
            zinit = Z[np.random.choice(len(Z))] + 0.0*(np.random.rand(model.hidden_size)-0.5)
            zi = pca.transform(zinit[None,:]); plt.plot(zi[:,0], zi[:,1], '.', markersize=1, alpha=0.2)

            h = torch.Tensor(zinit)[None,None,:]
            zs = []
            zs.append(h.detach().numpy().flatten())
            for _ in range(niters):
                a, (q, h) = model.sample_action(cobs, h.to(device), epsilon=0)
                zs.append(h.detach().numpy().flatten())
            zs = np.vstack(zs)

            if False:#sign <= 0:
                v = zs[1]-zs[0]
                z0 = pca.transform(np.vstack([zs[0], zs[0] + 1*v]))#/np.linalg.norm(v)]))
                plt.plot([z0[0,0], z0[1,0]], [z0[0,1], z0[1,1]], '-', color='k' if sign==0 else 'r', linewidth=1, alpha=0.2)
                plt.plot(z0[1,0], z0[1,1], 'k.', markersize=1, alpha=0.2)
            fps.append(zs[-1])
        
        fps = pca.transform(np.vstack(fps))
        h = plt.plot(fps[:,0], fps[:,1], '+', color=hf[0].get_color(), markersize=5, linewidth=1, label=name)

if showQ:
    plt.axis('equal')
    plt.plot(plt.xlim(), plt.xlim(), 'k-', alpha=0.5, linewidth=1, zorder=-1)
plt.xlabel('${}_1$'.format(lbl))
plt.ylabel('${}_2$'.format(lbl))
plt.legend(fontsize=8)

#%% belief fixed points

from analyze import belief_fixed_points_beron2022

Bs, b_inits = belief_fixed_points_beron2022(0.8, 0.02, niters=100)
doPlot = True

# regardless of p_rew_max and p_switch
# - there are two fixed points (call them xCoh and xIncoh) as long as 0.5 < p_rew_max ≤ 1
# - when p_rew_max = 0.5, there is one fixed point at 0.5
# - xCoh is the fixed point given a_prev == r_prev; xIncoh is the fixed point given a_prev != r_prev
# - xIncoh = 1-xCoh
# - when p_rew_max = 0.5, xCoh = 0.5
# - when p_rew_max = 1, xCoh = 1-p_switch = p_stay
# - for 0.5 ≤ p_rew_max ≤ 1, xCoh is sigmoid-like growth from 0.5 to p_stay

# todo: does the RNN learn the symmetry of a_prev == r_prev?
# - i.e., given inputs (a_prev, r_prev), does it learn to treat (0,0) the same as (1,1), and (0,1) the same as (1,0)?
# - for at least one model I (partially) trained, the answer seems to be NO

for a_prev in [0,1]:
    for r_prev in [0,1]:
        b_ends = []
        for b_init in b_inits:
            bs = Bs[(a_prev,r_prev,b_init)]
            b_ends.append(bs[-1])
        print('a={}, r={}: b={:0.3f} ± {:0.3f}'.format(a_prev, r_prev, np.mean(b_ends), np.std(b_ends)/np.sqrt(len(b_ends))))

if doPlot:
    plt.figure(figsize=(6,6)); c = 1
    for a_prev in [0,1]:
        for r_prev in [0,1]:
            plt.subplot(2,2,c); c += 1
            for b_init in b_inits:
                bs = Bs[(a_prev,r_prev,b_init)]
                plt.plot(bs)
            plt.title('a={}, r={}'.format(a_prev, r_prev))
            plt.xlabel('# time steps')
            plt.ylabel('Belief')
    plt.tight_layout()

#%% plot belief fixed points as a function of experiment params

p_switches = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]
p_rew_maxs = np.linspace(0.5,1,21)

for p_switch in p_switches:
    b_ends = []
    for p_rew_max in p_rew_maxs:
        Bs, b_inits = belief_fixed_points_beron2022(p_rew_max, p_switch, niters=100)
        b_end = Bs[(a_prev,r_prev,0.5)][-1]
        b_ends.append(b_end)
    b_ends = np.array(b_ends)
    plt.plot(p_rew_maxs, b_ends, '.-', label='$p_{switch}$=' + '{}'.format(p_switch))
plt.xlabel('p reward max')
plt.ylabel('Belief')
plt.legend(fontsize=8)

for p_rew_max in p_rew_maxs[:1]:
    b_ends = []
    for p_switch in p_switches:
        Bs, b_inits = belief_fixed_points_beron2022(p_rew_max, p_switch, niters=100)
        b_end = Bs[(a_prev,r_prev,0.5)][-1]
        b_ends.append(b_end)

#%% get Q weights using Least Squares (note: valid only when p_rew_max=1)

from analyze import lsql

env = Beron2022_TrialLevel(p_rew_max=1, p_switch=0.1)
# todo: wrap env using appropriate wrappers
obs = env.reset(seed=555)[0]

trials = []
for i in range(1000):
    obs = env.reset()[0]

    a = int(np.random.rand() < 0.5)

    next_obs, r, done, truncated, info = env.step(a)

    if i > 0:
        trials.append((np.where(obs)[0][0], a, r, np.where(next_obs)[0][0]))
    obs = next_obs

w = lsql(trials, gamma=0)
print(np.round(w,3)) # A, b, a, B
