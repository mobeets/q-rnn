import numpy as np
from plotting.base import plt
from matplotlib.patches import Rectangle, Polygon
import torch

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

def get_action(trial):
    # get action on each trial (accounting for a possible reward delay)
    # return trial.A[-1]
    return trial.A[np.where(trial.X[:,0] == 1)[0][0]] if trial.X[:,0].sum() == 1 else trial.A[-1]

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
            A = np.hstack([get_action(trial) for trial in trials])
            switchInds = np.hstack([0, np.where(np.diff(S) != 0)[0] + 1, len(S)+1])
            
            if showHighPort:
                # Y = np.hstack([trial.S[-1] == trial.A[-1] for trial in trials])
                Y = (S == A)
            else:
                # Y = np.hstack([False, np.hstack([trials[t+1].A[-1] != trials[t].A[-1] for t in range(len(trials)-1)])])
                Y = np.hstack([False, A[1:] != A[:-1]])

            As = []
            for i in range(len(switchInds)-1):
                if i == 0:
                    continue
                xPre = np.max([switchInds[i-1], switchInds[i]-tBefore])
                x0 = switchInds[i]
                xPost = np.min([switchInds[i+1], switchInds[i]+tAfter])

                a_pre = Y[xPre:x0]
                a_post = Y[x0:xPost]
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
        mus = np.vstack(values).mean(axis=0)
        ses = np.nanstd(np.vstack(values), axis=0)/np.sqrt(len(values))
        if len(values) > 1:
            for vs in values:
                plt.plot(xs, vs, 'k-', linewidth=1, alpha=0.5)
        verts = [*zip(xs, mus-ses), *zip(xs[::-1], mus[::-1]+ses[::-1])]
        plt.gca().add_patch(Polygon(verts, facecolor='0.9', edgecolor='0.5'))
        plt.plot(xs, mus, 'k-')

    for showHighPort in [True, False]:
        plt.subplot(1,2,-int(showHighPort)+2)
        plt.xlabel('Block Position')
        if showHighPort:
            plt.ylabel('P(high port)')
        else:
            plt.ylabel('P(switch)')
        if showHighPort:
            plt.ylim([-0.02, 1.02])
        plt.plot([0, 0], plt.ylim(), 'k:', zorder=-1, alpha=0.5)
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
        # either an abort trial or a no-response trial
        return '_'

def toWord(seq):
    if seq[0].lower() == 'l':
        return seq.replace('l', 'a').replace('L', 'A').replace('r', 'b').replace('R', 'B')
    elif seq[0].lower() == 'r':
        return seq.replace('l', 'b').replace('L', 'B').replace('r', 'a').replace('R', 'A')
    else:
        assert False

mouseWordOrder = ['AAA', 'aAA', 'AaA', 'aaA', 'AbB', 'aBB', 'aBA', 'ABA', 'abA', 'abB', 'aaB', 'ABB', 'AAa', 'AaB', 'AbA', 'aAa', 'aAB', 'AAB', 'Aaa', 'aBb', 'ABb', 'Aba', 'aba', 'aaa', 'aab', 'aBa', 'ABa', 'Aab', 'abb', 'AAb', 'aAb', 'Abb']

def plot_switching_by_symbol(AllTrials, doShow=True, wordOrder=None, modelBased=False, tau=None):
    """
    characterize switching probs given 'words', as in Fig. 2D of Beron et al. (2022)
    """
    words = [x+y+z for x in 'Aa' for y in 'AaBb' for z in 'AaBb']
    if wordOrder is not None:
        words = wordOrder
    counts = {word: (0,0) for word in words}
    wordLength = len(words[0])
    all_switches = []

    # counts per model
    for trials in AllTrials:
        symbs = [toSymbol(get_action(trial), trial.R[-1]) for trial in trials]
        switches = []
        for i in range(len(trials)-wordLength-1):
            ctrials = trials[i:(i+wordLength+1)]
            if len(set([trial.episode_index for trial in ctrials])) > 1:
                continue
            csymbs = symbs[i:(i+wordLength+1)]
            if '_' in csymbs: # ignore any sequence with an abort/no-response trial
                continue
            curWord = toWord(''.join(csymbs))
            if modelBased:
                curQ = ctrials[-1].Q[-1]
                lastAction = get_action(ctrials[-2])
                if tau is None:
                    modelAction = np.argmax(curQ)
                    switchProb = modelAction != lastAction
                else:
                    modelProbs = torch.softmax(torch.Tensor(curQ)/tau, dim=-1).numpy()
                    switchProb = 1 - modelProbs[lastAction]
            else:
                switchProb = float(csymbs[-1].upper() != csymbs[-2].upper())
            cur = (curWord[:-1], switchProb)
            switches.append(cur)

        for (word, switchProb) in switches:
            if word not in counts:
                counts[word] = (0,0)
            c,n = counts[word]
            counts[word] = (c + switchProb, n+1)
        all_switches.append(switches)

    # tally averages
    freqs = [(word, vals[0]/vals[1] if vals[1] > 0 else 0, vals[1]) for word, vals in counts.items()]
    freqs = [(word, p, np.sqrt(p*(1-p)/n,) if n > 0 else 0, n) for word,p,n in freqs] # add binomial SE
    if wordOrder is None:
        freqs = sorted(freqs, key=lambda x: x[1])
    xs = np.arange(len(freqs))
    grayOutInds = np.array([i for i,(x,y,z,n) in enumerate(freqs) if n <= 0]).astype(int)

    plt.figure(figsize=(8,2.2))
    for x, (_,p,se,n) in zip(xs, freqs):
        # plt.bar(x, n > 10, color='k', alpha=0.5)
        plt.bar(x, p, color='k', alpha=0.5 if se < 0.2 else 0.2)
        plt.plot([x,x], [p-se, p+se], 'k-', linewidth=1)
    plt.xticks(ticks=xs, labels=[x for x,y,z,n in freqs], rotation=90)
    for ind in grayOutInds:
        plt.gca().get_xticklabels()[ind].set_color('gray')
    plt.yticks([0,0.25,0.5,0.75,1])
    plt.xlim([-1, xs.max()+1])
    plt.xlabel('history')
    plt.ylabel('P(switch)')
    plt.tight_layout()
    if doShow:
        plt.show()
    return freqs, all_switches

def plot_decoding_weights(weights, std_errors, names, ylim=None, doShow=True):
    plt.figure(figsize=(len(names)/2,1.5))
    plt.plot(weights, '.')
    for i, (w, se) in enumerate(zip(weights, std_errors[:-1])):
        plt.plot([i,i], [w-se, w+se], 'k-', alpha=0.5, linewidth=1, zorder=-1)
    plt.plot(plt.xlim(), [0, 0], 'k-', alpha=0.3, linewidth=1, zorder=-2)
    plt.xticks(ticks=range(len(names)), labels=names, rotation=90)
    plt.yticks(ticks=[0, 1, 2])
    plt.ylabel('weight')
    if ylim is not None:
        plt.ylim(ylim)
    if doShow:
        plt.show()

def plot_decoding_weights_grouped(weights, std_errors, feature_params, doShow=True, title=None):
    plt.figure(figsize=(3,2.5))
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#9467bd', '#e377c2', '#bcbd22', '#17becf']
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
    # plt.yticks(ticks=[0, 1])
    plt.xticks(ticks=np.arange(0, max(feature_params.values())+1, 2)[1:])
    plt.xlabel('lag')
    plt.ylabel('weight')
    plt.legend(fontsize=8, loc='upper right')
    if title:
        plt.title(title)
    if doShow:
        plt.show()
    