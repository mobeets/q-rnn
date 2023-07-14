#%% imports

import os.path
import glob
import numpy as np
import torch
from model import DRQN
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

#%% random agent

env = Beron2022_TrialLevel(ntrials=150)

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

# Fig. 1B from Beron et al. (2022)
plt.figure(figsize=(6,1.5))
xs = np.arange(len(trials))
plt.plot(xs, trials[:,0], 'k-', linewidth=1, zorder=-1, alpha=0.5)
plt.scatter(xs, trials[:,1], s=1 + 3*trials[:,2])
plt.yticks(ticks=[0,1], labels=['left', 'right'])
plt.xlabel('Trial')

# %%
