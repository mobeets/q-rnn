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

#%% eval many models

import glob
import os.path
from session import eval_model
from plotting.behavior import plot_example_actions, plot_average_actions_around_switch, plot_switching_by_symbol, plot_decoding_weights_grouped
from analysis.decoding_beron import get_rnn_decoding_weights, get_mouse_decoding_weights, load_mouse_data
from plotting.behavior import plot_decoding_weights, mouseWordOrder

epsilon = None; tau = 0.000001
ntrials = 10000
fnms = glob.glob(os.path.join('data', 'models', '*granasoft*.json')) # 'grant': trial-level, 'granz': timestep-level; 'grans': H=2 timestep-evel; 'granasoft': H=10 trial-level w/ softmax
AllTrials = []
AllTrialsRand = []
for fnm in fnms[-1:]:
    Trials, Trials_rand, _, _ = eval_model(fnm, ntrials, epsilon, tau)
    AllTrials.append(Trials)
    AllTrialsRand.append(Trials_rand)

plot_average_actions_around_switch([Trials['test'] for Trials in AllTrials])
plot_switching_by_symbol([Trials['test'] for Trials in AllTrials], wordOrder=None)
plot_switching_by_symbol([Trials['test'] for Trials in AllTrials], wordOrder=mouseWordOrder)
feature_params = {
    'choice': 5, # choice history
    'reward': 5, # reward history
    'choice*reward': 5, # rewarded trials, aligned to action (original)
    '-choice*omission': 5, # unrewarded trials, aligned to action
    'b': 0 # belief history
}
weights, std_errors, names, lls = get_rnn_decoding_weights(AllTrials, feature_params)
plot_decoding_weights_grouped(weights, std_errors, feature_params)

if 'mouse_trials' not in vars():
    mouse_trials = load_mouse_data()
weights, std_errors, names, lls = get_mouse_decoding_weights(mouse_trials, feature_params)
plot_decoding_weights_grouped(weights, std_errors, feature_params)

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

ntrials = 10000
env_params = {'p_rew_max': 0.8, 'p_switch': 0.02, 'ntrials': ntrials}
env = Beron2022_TrialLevel(**env_params)
env = PreviousRewardWrapper(env)
env = PreviousActionWrapper(env, env.action_space.n)
agent = BeronBeliefAgent(env)

epsilon = 0.04

nreps = 10
AllTrials = []
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
    AllTrials.append(Trials)

# analyze belief agent
from plotting.behavior import mouseWordOrder
plot_switching_by_symbol([Trials['test'] for Trials in AllTrials], wordOrder=None)
plot_switching_by_symbol([Trials['test'] for Trials in AllTrials], wordOrder=mouseWordOrder)
plot_example_actions(Trials['test'])
plot_average_actions_around_switch([Trials['test'] for Trials in AllTrials])
feature_params = {
    'a': 5, # choice history
    'r': 5, # reward history
    'x': 5, # rewarded trials, aligned to action (original)
    'y': 5, # unrewarded trials, aligned to action
    'b': 0 # belief history
}
weights, std_errors, names, lls = get_rnn_decoding_weights(AllTrials, feature_params)
plot_decoding_weights_grouped(weights, std_errors, feature_params)

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
