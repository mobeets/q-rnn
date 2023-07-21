#%% imports

import os.path
import glob
import numpy as np
import torch
from model import DRQN
from tasks.beron2022 import Beron2022_TrialLevel
from train import probe_model, probe_model_off_policy
from analyze import add_beliefs_beron2022
from analysis.correlations import analyze
device = torch.device('cpu')

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.figsize'] = [3.0, 3.0]
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

#%% example random agent

env = Beron2022_TrialLevel()
seed = 555
last_obs = env.reset(seed=seed)[0]
done = False
trials = []
while not done:
    obs = last_obs
    a = int(np.random.rand() < 0.5)
    last_obs, r, done, truncated, info = env.step(a)
    trials.append((info['state'], a, r))
trials = np.vstack(trials)

#%% load model

nepisodes = 1; ntrials_per_episode = 1000
# infile = 'data/models/weights_final_h3_beron_v3_p08.pth'
# infile = 'data/models/weights_final_h3_beron_v5_p08_softmax.pth'
# infile = 'data/models/weights_final_h3_beron_v5_p08_epsilon.pth'
# infile = 'data/models/weights_final_h3_beron_v6_p08_epsilon.pth'
# infile = 'data/models/weights_final_h2_beron_v7_p08_epsilon.pth'
# infile = 'data/models/weights_final_h6_beron_v8_p08_epsilon.pth'
# infile = 'data/models/weights_final_h6_beron_v9_p08_epsilon.pth'
# infile = 'data/models/weights_final_h2_beron_v10_p08_epsilon.pth'
# infile = 'data/models/weights_final_h2_beron_v11_p08_epsilon.pth'
# infile = 'data/models/weights_final_h2_beron_v12_p08_epsilon.pth'
infile = 'data/models/weights_final_h3_beron_v13_p1_epsilon.pth'

env_params = {'p_rew_max': float(infile.split('_p')[1].split('_')[0])/10}
hidden_size = int(infile.split('_h')[1].split('_')[0])
print('H={}, p={}'.format(hidden_size, env_params['p_rew_max']))

epsilon = 0.01; tau = None
# tau = 0.00001; epsilon = None

env = Beron2022_TrialLevel(**env_params)
model = DRQN(input_size=4, # empty + prev reward + prev actions
                hidden_size=hidden_size,
                output_size=env.action_space.n).to(device)
model.load_weights_from_path(infile)

policymodel = DRQN(input_size=4, # empty + prev reward + prev actions
                hidden_size=hidden_size,
                output_size=env.action_space.n).to(device)
policymodel.load_weights_from_path(infile)
# policymodel = None # purely random policy

# probe model
Trials = {}
Trials_rand = {}
for useRandomModel in [True, False]:
    if useRandomModel:
        model.reset(gain=1)
    else:
        model.load_weights_from_path(infile)
    
    for name, seed in {'train': 456, 'test': 787}.items():
        # reset seeds
        env.state = None
        env.reset(seed=seed)
        model.reset_rng(seed+1)
        if policymodel is not None:
            policymodel.reset_rng(seed+2)

        # run model on trials
        trials = probe_model_off_policy(model, policymodel, env,
                                        epsilon=epsilon, tau=tau,
                                        nepisodes=nepisodes, ntrials_per_episode=ntrials_per_episode)
        print(useRandomModel, name, np.hstack([trial.R for trial in trials]).mean())

        # add beliefs
        add_beliefs_beron2022(trials, env.p_rew_max, env.p_switch)
        
        # discard trials at the beginning of episodes (warmup)
        trials = [trial for trial in trials if trial.index_in_episode > 5]

        if useRandomModel:
            Trials_rand[name] = trials
        else:
            Trials[name] = trials

#%% plot Fig. 1B from Beron et al. (2022)

from matplotlib.patches import Rectangle

plt.figure(figsize=(9,1.5))

trials = Trials['test']
# env.reset(seed=seed); model.reset_rng(seed+1); env.state = None
# trials = probe_model(model, env, nepisodes=1, ntrials_per_episode=1000, epsilon=0.05)
# add_beliefs_beron2022(trials, env.p_rew_max, env.p_switch)
# trials = [trial for trial in trials if trial.index_in_episode > 3]

S = np.vstack([trial.S for trial in trials])[:,0]
A = np.vstack([trial.A for trial in trials])[:,0]
R = np.vstack([trial.R for trial in trials])[:,0]
B = np.vstack([trial.B for trial in trials])[:,0]
xs = np.arange(len(S))

switchInds = np.hstack([0, np.where(np.diff(S) != 0)[0] + 1, len(S)+1])
for i in range(len(switchInds)-1):
    x1 = switchInds[i]
    x2 = switchInds[i+1]
    if S[x1] == 1:
        rect = Rectangle((x1-0.5, -0.05), x2-x1, 1.1, alpha=0.4)
        plt.gca().add_patch(rect)

# plt.plot(xs, B, 'k-', linewidth=1, label='Beliefs')
plt.scatter(xs, A, s=1+5*R, c='k')
plt.yticks(ticks=[0,1], labels=['left', 'right'])
plt.xlabel('Trial')
plt.xlim(0 + np.array([0, 140.5]))

#%% plot Fig. 1C/D from Beron et al. (2022)

tBefore = 10
tAfter = 20

trials = Trials['test']
# env.reset(seed=seed); model.reset_rng(seed+1); env.state = None
# trials = probe_model(model, env, nepisodes=1, ntrials_per_episode=1000, epsilon=0.05)
# add_beliefs_beron2022(trials, env.p_rew_max, env.p_switch)
# trials = [trial for trial in trials if trial.index_in_episode > 3]

S = np.vstack([trial.S for trial in trials])[:,0]
switchInds = np.hstack([0, np.where(np.diff(S) != 0)[0] + 1, len(S)+1])

plt.figure(figsize=(6,2))
for showHighPort in [True, False]:
    plt.subplot(1,2,-int(showHighPort)+2)
    if showHighPort:
        A = np.vstack([trial.S == trial.A for trial in trials])[:,0]
    else:
        A = np.hstack([False, np.vstack([trials[t+1].A != trials[t].A for t in range(len(trials)-1)])[:,0]])

    As = []
    for i in range(len(switchInds)-1):
        if i == 0:
            continue
        xPre = np.max([switchInds[i-1], switchInds[i]-tBefore])
        x0 = switchInds[i]
        xPost = np.min([switchInds[i+1], switchInds[i]+tAfter])

        a_pre = A[xPre:x0]
        a_post = A[x0:xPost]
        if len(a_pre) < tBefore:
            n = tBefore - len(a_pre)
            a_pre = np.hstack([np.nan*np.ones(n), a_pre])
        if len(a_post) < tAfter:
            n = tAfter - len(a_post)
            a_post = np.hstack([a_post, np.nan*np.ones(n)])
        ac = np.hstack([a_pre, a_post])
        As.append(ac)

    As = np.vstack(As)
    xs = np.arange(-tBefore, tAfter)

    plt.plot(xs, np.nanmean(As, axis=0), 'k.-')
    plt.plot([0, 0], [-0.05, 1.05], 'k--', zorder=-1, alpha=0.5)
    plt.xlabel('Block Position')
    if showHighPort:
        plt.ylabel('P(high port)')
    else:
        plt.ylabel('P(switch)')
    plt.ylim([-0.02, 1.02])
plt.tight_layout()

#%% characterize switching probs given 'words', as in Fig. 2D of Beron et al. (2022)

def toSymbol(a,r):
    if a == 0 and r == 0:
        return 'l'
    elif a == 0 and r == 1:
        return 'L'
    elif a == 1 and r == 0:
        return 'r'
    elif a == 1 and r == 1:
        return 'R'
    else:
        assert False

def toWord(seq):
    if seq[0].lower() == 'l':
        return seq.replace('l', 'a').replace('L', 'A').replace('r', 'b').replace('R', 'B')
    elif seq[0].lower() == 'r':
        return seq.replace('l', 'b').replace('L', 'B').replace('r', 'a').replace('R', 'A')
    else:
        assert False

trials = Trials['test']
trials = probe_model(model, env, nepisodes=50, ntrials_per_episode=1000, epsilon=0.01)

symbs = [toSymbol(trial.A[0], trial.R[0]) for trial in trials]
switches = []
for i in range(len(trials)-4):
    ctrials = trials[i:(i+4)]
    if any([trial.index_in_episode <= 3 for trial in ctrials]):
        continue
    if len(set([trial.episode_index for trial in ctrials])) > 1:
        continue
    cur = (toWord(''.join(symbs[i:(i+3)])), trials[i+4].A[0] != trials[i+3].A[0])
    switches.append(cur)

words = [x+y+z for x in 'Aa' for y in 'AaBb' for z in 'AaBb']
counts = {word: (0,0) for word in words}
for (word, didSwitch) in switches:
    if word not in counts:
        counts[word] = (0,0)
    c,n = counts[word]
    counts[word] = (c + int(didSwitch), n+1)
freqs = [(word, vals[0]/vals[1] if vals[1] > 0 else 0, vals[1]) for word, vals in counts.items()]
freqs = [(word, p, np.sqrt(p*(1-p)/n) if n > 0 else 0) for word,p,n in freqs] # add binomial SE
freqs = sorted(freqs, key=lambda x: x[1])
xs = np.arange(len(freqs))

plt.figure(figsize=(8,2))
plt.bar(xs, [y for x,y,z in freqs], color='k', alpha=0.5)
for x, (_,p,se) in zip(xs, freqs):
    plt.plot([x,x], [p-se, p+se], 'k-', linewidth=1)
plt.xticks(ticks=xs, labels=[x for x,y,z in freqs], rotation=90)
plt.yticks([0,0.25,0.5,0.75,1])
plt.xlim([-1, xs.max()+1])
plt.xlabel('history')
plt.ylabel('P(switch)')

#%% compare beliefs and latent activity

results_rand = analyze(Trials_rand, key='Z')
# print(results_rand['rsq'])
results = analyze(Trials, key='Z')
# print(results['rsq'])

plt.figure(figsize=(1,2))
plt.plot([1,2], [results_rand['rsq'], results['rsq']], 'ko')
plt.xlim([0.5, 2.5])
plt.xticks(ticks=[1,2], labels=['Untrained\nRQN', 'RQN'], rotation=90)
plt.ylim([-0.05,1.05])

#%% plot belief predictions over trials

trials = Trials['test']
S = np.vstack([trial.S for trial in trials])[:,0]
# A = np.vstack([trial.A for trial in trials])[:,0]
# R = np.vstack([trial.R for trial in trials])[:,0]
B = np.vstack([trial.B for trial in trials])[:,0]
Bhat = np.vstack([trial.Bhat for trial in trials])
Q = np.vstack([trial.Q for trial in trials])
Qdiff = Q[:,1] - Q[:,0]; Qdiff /= np.abs(Qdiff).max(); Qdiff /= 2; Qdiff += 0.5

from matplotlib.patches import Rectangle

plt.figure(figsize=(9,1.5))
xs = np.arange(len(trials))

S = np.vstack([trial.S for trial in trials])[:,0]
A = np.vstack([trial.A for trial in trials])[:,0]
R = np.vstack([trial.R for trial in trials])[:,0]

switchInds = np.hstack([0, np.where(np.diff(S) != 0)[0] + 1, len(S)+1])
for i in range(len(switchInds)-1):
    x1 = switchInds[i]
    x2 = switchInds[i+1]
    if S[x1] == 1:
        rect = Rectangle((x1-0.5, -0.05), x2-x1, 1.1, alpha=0.3)
        plt.gca().add_patch(rect)

plt.plot(xs, B, 'k-', linewidth=1, label='Beliefs')
plt.plot(xs, Bhat, 'r-', linewidth=1, alpha=0.6, label='$\widehat{B}$')
plt.plot(xs, Qdiff, 'b-', linewidth=1, alpha=0.6, label='$\Delta Q$')
plt.yticks(ticks=[0,1], labels=['left', 'right'])
plt.xlabel('Trial')
plt.xlim(0 + np.array([0, 140.5]))
plt.legend(fontsize=9)

#%% compare belief predictions

plt.plot(Bhat, B, 'r.', markersize=5, alpha=0.6, label='$\widehat{B}$')
plt.plot(Qdiff, B, 'b.', markersize=5, alpha=0.6, label='$\Delta Q$')
plt.xlabel('Belief prediction')
plt.ylabel('Belief')
plt.xticks([0,0.5,1]); plt.yticks([0,0.5,1])
plt.legend(fontsize=9)

#%% visualize activity

from analysis.pca import fit_pca, apply_pca

pca = fit_pca(Trials['train'][10:])
trials = apply_pca(Trials['test'][10:], pca)

Z = np.vstack([trial.Z_pc for trial in trials])
# Z = np.vstack([trial.Z for trial in trials])
# Z = np.vstack([trial.Q for trial in trials])

S = np.vstack([trial.S for trial in trials])[:,0]
A = np.vstack([trial.A for trial in trials])[:,0]
R = np.vstack([trial.R for trial in trials])[:,0]

switchInds = np.hstack([0, np.where(np.diff(S) != 0)[0] + 1, len(S)+1])
for i in range(len(switchInds)-1):
    x1 = switchInds[i]
    x2 = switchInds[i+1]
    if x2 - x1 < 10:
        continue
    zc = Z[x1:x2]
    if S[x1] == 1:
        color = 'b'
    else:
        color = 'r'
    plt.plot(zc[0,0], zc[0,1], '+', markersize=5, color=color, zorder=1)
    plt.plot(zc[:,0], zc[:,1], '.-', markersize=1, linewidth=1, color=color, alpha=0.2, zorder=0)
    plt.plot(zc[-1,0], zc[-1,1], 'v', markersize=5, color=color, zorder=1)

#%% assess RNN responses to fixed inputs

from analysis.pca import fit_pca, apply_pca

ninits = 100
niters = 100
showFPs = True

pca = fit_pca(Trials['train'][10:])
# pca.transform = lambda x: x
trials = apply_pca(Trials['test'][10:], pca)

X = np.vstack([trial.X for trial in trials])
Z = np.vstack([trial.Z for trial in trials])
Zpc = pca.transform(Z)
# Zpc = np.vstack([trial.Q for trial in trials]); showFPs = False

zmin = Z.min()-0.01
zmax = Z.max()+0.01

plt.figure(figsize=(3,3))
# plt.plot(Z[:,0], Z[:,1], 'k.', markersize=1, alpha=0.1, zorder=-1)

for sign in [0,1,2,3,4,5]:
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
    if sign < 4:
        name = 'r={}, a={}'.format(r_prev, np.where(a_prev)[0][0])
    else:
        name = 'r={}'.format(r_prev)

    obs = np.hstack([[0], [r_prev], a_prev])
    cobs = torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0)
    ix = np.all(X == obs, axis=1)
    hf = plt.plot(Zpc[ix,0], Zpc[ix,1], '.', markersize=1, alpha=0.3, zorder=-1, label=name if not showFPs else '_')

    if showFPs:
        fps = []
        for i in range(ninits):
            zinit = np.random.rand(model.hidden_size)*(zmax-zmin) + zmin
            h = torch.Tensor(zinit)[None,None,:]
            zs = []
            zs.append(h.detach().numpy().flatten())
            for _ in range(niters):
                a, (q, h) = model.sample_action(cobs, h.to(device), epsilon=0)
                zs.append(h.detach().numpy().flatten())
            zs = np.vstack(zs)
            # plt.plot(zs[:,0], zs[:,1], '.-', markersize=1, linewidth=1, alpha=0.2)
            fps.append(zs[-1])
        
        fps = pca.transform(np.vstack(fps))
        h = plt.plot(fps[:,0], fps[:,1], '+', color=hf[0].get_color(), markersize=5, linewidth=1, label=name)

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

#%% choice regression (using same code as for mice)

from analysis.decoding_beron import fit_logreg_policy, compute_logreg_probs

if True:#'rnn_features' not in vars() or 'train' not in rnn_features:
    rnn_features = {}
    for name in ['train', 'test']:
        # trials = Trials[name]
        trials = probe_model(model, env, nepisodes=1, ntrials_per_episode=10000, epsilon=0.0)
        A = np.vstack([trial.A for trial in trials])[:,0]
        R = np.vstack([trial.R for trial in trials])[:,0]
        rnn_features[name] = [[A,R]]

pm1 = lambda x: 2 * x - 1
feature_functions = [
    lambda cs, rs: pm1(cs),                # choices
    lambda cs, rs: rs,                     # rewards
    lambda cs, rs: pm1(cs) * rs,           # +1 if choice=1 and reward, 0 if no reward, -1 if choice=0 and reward
    lambda cs, rs: -pm1(cs) * (1-rs),      # -1 if choice=1 and no reward, 1 if reward, +1 if choice=0 and no reward
    lambda cs, rs: pm1((cs == rs)),   
    lambda cs, rs: np.ones(len(cs))        # overall bias term
]

feature_params = {
    'A': 5, # choice history
    'R': 0, # reward history
    'A*R': 5, # rewarded trials, aligned to action (original)
    'A*(R-1)': 5, # unrewarded trials, aligned to action
    'B': 0 # belief history
}
memories = [y for x,y in feature_params.items()] + [1]
names = ['{}(t-{})'.format(name, t) for name, ts in feature_params.items() for t in range(ts)]

lr = fit_logreg_policy(rnn_features['train'], memories, feature_functions) # refit model with reduced histories, training set
model_probs, lls, std_errors = compute_logreg_probs(rnn_features['test'], [lr, memories], feature_functions)

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
