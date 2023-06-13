import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.figsize'] = [3.0, 3.0]
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

def plot_results(all_trials, ntrials_per_episode, outfile=None):
    fontsize = 8
    running_mean = lambda ys, n: np.convolve(ys, np.ones(n)/n, mode='valid')
    trials = np.vstack(all_trials)
    ylbls = ['avg reward', 'avg trial length', '# aborts', 'pct. correct']
    plt.figure(figsize=(3,3))
    for c, ylbl in enumerate(ylbls):
        plt.subplot(2,2,c+1)
        plt.xlabel('# episodes', fontsize=fontsize)
        plt.ylabel(ylbl, fontsize=fontsize)
        if ylbl == 'avg reward':
            ys = trials[:,-3]
        elif ylbl == 'avg trial length':
            ys = trials[:,0]
        elif ylbl == '# aborts':
            ys = trials[:,-2]
        elif ylbl == 'pct. correct':
            ys = trials[:,-1]
        plt.plot(running_mean(ys, ntrials_per_episode)[::ntrials_per_episode])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
        plt.close()
