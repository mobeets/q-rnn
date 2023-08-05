import numpy as np
from plotting.base import plt
from matplotlib.patches import Rectangle

def plot_example_actions(trials, doShow=True):
    """
    plot Fig. 1B from Beron et al. (2022)
    """
    plt.figure(figsize=(9,1.5))

    S = np.hstack([trial.S[-1] for trial in trials])
    A = np.hstack([trial.A[-1] for trial in trials])
    R = np.hstack([trial.R[-1] for trial in trials])
    xs = np.arange(len(S))

    switchInds = np.hstack([0, np.where(np.diff(S) != 0)[0] + 1, len(S)+1])
    for i in range(len(switchInds)-1):
        x1 = switchInds[i]
        x2 = switchInds[i+1]
        if S[x1] == 1:
            rect = Rectangle((x1-0.5, -0.05), x2-x1, 1.1, alpha=0.4)
            plt.gca().add_patch(rect)

    plt.scatter(xs, A, s=1+5*R, c='k')
    plt.yticks(ticks=[0,1], labels=['left', 'right'])
    plt.xlabel('Trial')
    plt.xlim(0 + np.array([0, 140.5]))
    plt.tight_layout()
    if doShow:
        plt.show()

def plot_average_actions_around_switch(AllTrials, tBefore=10, tAfter=20, doShow=True):
    """
    plot Fig. 1C-D from Beron et al. (2022)
    """
    plt.figure(figsize=(6,2))
    
    for showHighPort in [True, False]:
        plt.subplot(1,2,-int(showHighPort)+2)
        values = []
        for trials in AllTrials:
            S = np.hstack([trial.S[-1] for trial in trials])
            switchInds = np.hstack([0, np.where(np.diff(S) != 0)[0] + 1, len(S)+1])
            
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
            values.append(np.nanmean(As, axis=0))

        xs = np.arange(-tBefore, tAfter)
        if len(values) > 1:
            for vs in values:
                plt.plot(xs, vs, 'k-', linewidth=1, alpha=0.25)
        mus = np.vstack(values).mean(axis=0)
        plt.plot(xs, mus, 'k-')

    for showHighPort in [True, False]:
        plt.subplot(1,2,-int(showHighPort)+2)
        plt.plot([0, 0], [-0.05, 1.05], 'k:', zorder=-1, alpha=0.5)
        plt.xlabel('Block Position')
        if showHighPort:
            plt.ylabel('P(high port)')
        else:
            plt.ylabel('P(switch)')
        plt.ylim([-0.02, 1.02])
        if not showHighPort:
            plt.ylim([-0.02, 0.5])
        plt.xlim([-tBefore, tAfter])
    plt.tight_layout()
    if doShow:
        plt.show()

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

def plot_switching_by_symbol(AllTrials, doShow=True):
    """
    characterize switching probs given 'words', as in Fig. 2D of Beron et al. (2022)
    """
    words = [x+y+z for x in 'Aa' for y in 'AaBb' for z in 'AaBb']
    counts = {word: (0,0) for word in words}

    # counts per model
    for trials in AllTrials:
        symbs = [toSymbol(trial.A[-1], trial.R[-1]) for trial in trials]
        switches = []
        for i in range(len(trials)-4):
            ctrials = trials[i:(i+4)]
            if len(set([trial.episode_index for trial in ctrials])) > 1:
                continue
            cur = (toWord(''.join(symbs[i:(i+3)])), trials[i+4].A[0] != trials[i+3].A[0])
            switches.append(cur)

        for (word, didSwitch) in switches:
            if word not in counts:
                counts[word] = (0,0)
            c,n = counts[word]
            counts[word] = (c + int(didSwitch), n+1)

    # tally averages
    freqs = [(word, vals[0]/vals[1] if vals[1] > 0 else 0, vals[1]) for word, vals in counts.items()]
    freqs = [(word, p, np.sqrt(p*(1-p)/n) if n > 0 else 0) for word,p,n in freqs] # add binomial SE
    freqs = sorted(freqs, key=lambda x: x[1])
    xs = np.arange(len(freqs))

    plt.figure(figsize=(9,2))
    for x, (_,p,se) in zip(xs, freqs):
        plt.bar(x, p, color='k', alpha=0.5 if se < 0.2 else 0.2)
        plt.plot([x,x], [p-se, p+se], 'k-', linewidth=1)
    plt.xticks(ticks=xs, labels=[x for x,y,z in freqs], rotation=90)
    plt.yticks([0,0.25,0.5,0.75,1])
    plt.xlim([-1, xs.max()+1])
    plt.xlabel('history')
    plt.ylabel('P(switch)')
    plt.tight_layout()
    if doShow:
        plt.show()

def plot_decoding_weights(weights, std_errors, names, doShow=True):
    plt.figure(figsize=(len(names)/2,1.5))
    plt.plot(weights, '.')
    for i, (w, se) in enumerate(zip(weights, std_errors[:-1])):
        plt.plot([i,i], [w-se, w+se], 'k-', alpha=0.5, linewidth=1, zorder=-1)
    plt.plot(plt.xlim(), [0, 0], 'k-', alpha=0.3, linewidth=1, zorder=-2)
    plt.xticks(ticks=range(len(names)), labels=names, rotation=90)
    plt.yticks(ticks=[0, 1, 2])
    plt.ylabel('weight')
    plt.ylim([-0.4,2.3])
    if doShow:
        plt.show()

def plot_decoding_weights_grouped(weights, std_errors, feature_params, doShow=True):
    plt.figure(figsize=(3,2.5))
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    i = 0
    for c, (name, v) in enumerate(feature_params.items()):
        if v == 0:
            continue
        ws = weights[i:(i+v)]
        ses = std_errors[i:(i+v)]
        color = colors[c]
        h = plt.plot(np.arange(len(ws)) + 1, ws, '-' if len(ws) > 1 else '.', color=color, label=name, zorder=1)
        for j, (w, se) in enumerate(zip(ws, ses)):
            plt.plot(j*np.ones(2) + 1, [w-se, w+se], '-', color=h[0].get_color(), alpha=0.5, linewidth=1, zorder=0)
        i += v
    plt.xlim([0.9, max(feature_params.values())+0.1])
    plt.plot(plt.xlim(), np.zeros(2), 'k-', linewidth=1, alpha=0.5, zorder=-1)
    plt.yticks(ticks=[0, 1, 2])
    plt.xlabel('lag')
    plt.ylabel('weight')
    plt.legend(fontsize=8)
    if doShow:
        plt.show()
    
