#%% imports

import os.path
import glob
import json
import numpy as np
import torch
from model import DRQN
from tasks.beron2022 import Beron2022, Beron2022_TrialLevel, BeronCensorWrapper, BeronWrapper
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
run_name = 'h2_berontime3'
run_name = 'h3_berontime4'
run_name = 'h3_berontime5'
# run_name = 'h3_berontimep1'
run_name = 'h3_berontime6'
run_name = 'h3_berontime7'
run_name = 'h3_brnlambda'
run_name = 'h3_brnlambda2'

args = json.load(open('data/models/results_{}.json'.format(run_name)))
env_params = {
    'p_rew_max': args.get('p_reward_max', 0.8),
    'p_switch': args.get('p_switch', 0.02),
    'iti_min': args.get('iti_min', 0), 'iti_p': args.get('iti_p', 0.5), 
    'abort_penalty': args.get('abort_penalty', 0),
    'include_null_action': args.get('abort_penalty', 0) < 0}
hidden_size = args['hidden_size']
modelfile = args['filenames']['weightsfile_final']
initial_modelfile = args['filenames']['weightsfile_initial']
print('H={}, prew={}, pswitch={}'.format(hidden_size, env_params['p_rew_max'], env_params['p_switch']))
input_size = 1 + args['include_prev_reward'] + args['include_prev_action']*env.action_space.n
if args['experiment'] == 'beron2022_time':
    input_size += args.get('include_beron_wrapper', False)

epsilon = 0; tau = None
# tau = 0.00001; epsilon = None
nepisodes = 1; ntrials_per_episode = 1000

if args['experiment'] == 'beron2022_time':
    env = Beron2022(**env_params)
else:
    env = Beron2022_TrialLevel(**env_params)
if args.include_prev_reward:
    env = PreviousRewardWrapper(env)
if args.include_prev_action:
    env = PreviousActionWrapper(env, env.action_space.n)
if args.include_beron_wrapper:
    env = BeronWrapper(env, input_size)
if args.include_beron_censor:
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
                                epsilon=epsilon, tau=tau,
                                nepisodes=nepisodes, ntrials_per_episode=ntrials_per_episode)
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

#%% plot Fig. 1B from Beron et al. (2022)

from matplotlib.patches import Rectangle

plt.figure(figsize=(9,1.5))

trials = Trials['test']
S = np.hstack([trial.S[-1] for trial in trials])
A = np.hstack([trial.A[-1] for trial in trials])
R = np.hstack([trial.R[-1] for trial in trials])
B = np.hstack([trial.B[-1] for trial in trials])
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

S = np.hstack([trial.S[-1] for trial in trials])
switchInds = np.hstack([0, np.where(np.diff(S) != 0)[0] + 1, len(S)+1])

plt.figure(figsize=(6,2))
for showHighPort in [True, False]:
    plt.subplot(1,2,-int(showHighPort)+2)
    if showHighPort:
        A = np.hstack([trial.S[-1] == trial.A[-1] for trial in trials])
    else:
        A = np.hstack([False, np.hstack([trials[t+1].A[-1] != trials[t].A[-1] for t in range(len(trials)-1)])])

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

symbs = [toSymbol(trial.A[-1], trial.R[-1]) for trial in trials]
switches = []
for i in range(len(trials)-4):
    ctrials = trials[i:(i+4)]
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
# print(results_rand['rsq'])
results = analyze(Trials, key='Z')
resq = analyze(Trials, key='Qdiff')
# print(results['rsq'])

ys = [results_rand['rsq'], results['rsq'], resq['rsq']]
plt.figure(figsize=(0.5*len(ys),2))
plt.plot(range(len(ys)), ys, 'ko')
plt.xlim([-0.5, len(ys)-0.5])
plt.xticks(ticks=range(len(ys)), labels=['Untrained\nRQN', 'RQN', 'RQN-ΔQ'], rotation=90)
plt.ylim([-0.05,1.05])

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
showQ = False

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

#%% choice regression (using same code as for mice)

from analysis.decoding_beron import fit_logreg_policy, compute_logreg_probs

if True:#'rnn_features' not in vars() or 'train' not in rnn_features:
    rnn_features = {}
    for name in ['train', 'test']:
        trials = Trials[name]
        A = np.hstack([trial.A[-1] for trial in trials])
        R = np.hstack([trial.R[-1] for trial in trials])
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
names = ['{}(t-{})'.format(name, t+1) for name, ts in feature_params.items() for t in range(ts)]

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
