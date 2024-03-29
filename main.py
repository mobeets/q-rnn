#%% imports

import os.path
import glob
import numpy as np
import torch
from model import DRQN
from tasks.roitman2002 import Roitman2002
from tasks.beron2022 import Beron2022_TrialLevel
from train import probe_model, probe_model_off_policy
from analyze import add_beliefs
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

#%% load model

nepisodes = 1; ntrials_per_episode = 1000
seed = 666
infile = 'data/models/weights_final_h3_beron_v3_p09.pth'
env_params = {'p_rew_max': 0.9, 'ntrials': 1}
env = Beron2022_TrialLevel(**env_params)
hidden_size = 3
policymodel = DRQN(input_size=4, # empty + prev reward + prev actions
                hidden_size=hidden_size,
                output_size=env.action_space.n).to(device)
policymodel.load_weights_from_path(infile)
model = policymodel

# probe model
env.reset(seed=seed)
trials = probe_model(model, env, nepisodes=nepisodes, ntrials_per_episode=ntrials_per_episode)
pol_avg_rew = np.hstack([trial.R for trial in trials]).mean()

#%% compare belief-like-ness of models, before and after training

seed = 666
nepisodes = 500; ntrials_per_episode = 1
reward_amounts = [20, -400, -400, -1]
env = Roitman2002(reward_amounts=reward_amounts)

Results = {}
for infile in glob.glob('data/weights_final_*.pth')[:1]:
    hidden_size = int(infile.split('_h')[1].split('_')[0])
    policymodel = DRQN(input_size=2, # stim and reward
                    hidden_size=hidden_size,
                    output_size=env.action_space.n).to(device)
    policymodel.load_weights_from_path(infile)

    for mode in ['initial', 'final']:
        cur_infile = infile.replace('_final_', '_{}_'.format(mode))
        model = DRQN(input_size=2, # stim and reward
                    hidden_size=hidden_size,
                    output_size=env.action_space.n).to(device)
        model.load_weights_from_path(cur_infile)
        # todo: add previous action

        # probe model
        env.reset(seed=seed)
        trials = probe_model_off_policy(model, policymodel, env, nepisodes=nepisodes, ntrials_per_episode=ntrials_per_episode)
        pol_avg_rew = np.hstack([trial.R for trial in trials]).mean()
        
        env.reset(seed=seed)
        trials_onpolicy = probe_model(model, env, nepisodes=nepisodes, ntrials_per_episode=ntrials_per_episode)
        avg_rew = np.hstack([trial.R for trial in trials_onpolicy]).mean()

        # add beliefs
        B, (O, T) = add_beliefs(trials, p_iti=env.iti_p, p_coh=env.p_coh)

        # belief regression
        nTrialsTrain = int(len(trials)/2)
        Trials = {'train': trials[:nTrialsTrain], 'test': trials[nTrialsTrain:]}
        Results[(cur_infile, mode)] = analyze(None, Trials)
        print(cur_infile, mode, pol_avg_rew, avg_rew, Results[(cur_infile, mode)]['rsq'])

#%% visualize successful trial

for trial in trials:
    trial.tag = None
    if trial.trial_length < trial.iti:
        trial.tag = 'aborted'
    elif np.any(trial.R == env.reward_amounts[0]):
        trial.tag = 'correct'
    elif np.any(trial.R == env.reward_amounts[1]) and trial.trial_length >= trial.iti:
        trial.tag = 'incorrect'
    assert trial.tag is not None

for trial in trials[1:]:
    if trial.state == 1:
        if trial.tag == 'correct':
            plt.plot(2.5*trial.X[:,0], '.', linewidth=1)
            plt.plot(trial.Q, '.-', linewidth=1)
            break
