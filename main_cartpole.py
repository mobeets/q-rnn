#%% imports

import numpy as np
from plotting.base import plt
import gymnasium as gym
from ray.rllib.algorithms import r2d2
from ray.tune.registry import register_env
from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole
from tasks.cartpole import DelayedStatelessCartpole

#%% initialize model

env_config = {'delay': 0}
env_name = 'stateless-cartpole'
register_env(env_name, lambda env_config: DelayedStatelessCartpole(**env_config))

config = r2d2.R2D2Config()
config = config.environment(env_name, env_config=env_config)
# config.rollouts(num_rollout_workers=0)
config.lr = 0.0005
# config.gamma = 0.9 # 0.997
config.exploration_config['initial_epsilon'] = 1.0
config.exploration_config['final_epsilon'] = 0.0
config.exploration_config['epsilon_timesteps'] = 50000 # divide by 1000 to get number of batches
# config.dueling = False
# config.hiddens = []
config.model['use_lstm'] = True
config.model['lstm_cell_size'] = 64
config.model['lstm_use_prev_action'] = True
config.model['max_seq_len'] = 20
config.model['fcnet_hiddens'] = [64]
config.model['fcnet_activation'] = 'linear'
config.replay_buffer_config['replay_burn_in'] = 20
config.evaluation_duration = 10; config.evaluation_interval = 1 # evaluate using 10 episodes every episode

algo = config.build()
print(algo.get_policy().model)

#%% train model

outputs = []
best_score = 0
checkpoints = []

get_score = lambda x: x['evaluation']['episode_reward_mean'] if 'evaluation' in x else x['episode_reward_mean']
for i in range(300):
    output = algo.train()
    outputs.append(output)

    score = get_score(outputs[-1])
    epsilons = algo.workers.foreach_worker(lambda worker: worker.get_policy().exploration.get_state()['cur_epsilon'])
    print(i, score, epsilons)

    if score > best_score:
        # note: this isn't always the best model...
        best_score = score
        checkpoint_dir = algo.save()
        checkpoints.append(checkpoint_dir)
        print("Saved checkpoint to {}.".format(checkpoint_dir))

checkpoints.append(algo.save())
print('Best score: {}\n    Checkpoint: '.format(best_score, checkpoints[-2]))
# algo.restore(checkpoints[-2]) # restore best scoring model

plt.plot([get_score(x) for x in outputs])

#%% plot results

plt.plot([get_score(x) for x in outputs])

#%% rollout

algo.restore(checkpoints[-2])
delays = list(np.arange(6))
# delays = [env_config['delay']]

randomPolicy = True
explore = False
nepisodes = 100
perfs = []
AllTrials = {}

for delay in delays:
    env = DelayedStatelessCartpole(**{'delay': delay})

    Trials = []
    for j in range(nepisodes):
        terminated = truncated = False
        obs, info = env.reset()

        state = algo.get_policy().get_initial_state()
        action = 0
        reward = 0

        trials = []
        i = 0
        while not terminated and not truncated:
            action, state, q_info = algo.compute_single_action(obs,
                    prev_action=action if config.model['lstm_use_prev_action'] else None,
                    state=state, explore=explore)
            if randomPolicy:
                action = int(np.random.rand() < 0.5)
            new_obs, reward, terminated, truncated, info = env.step(action)
            trials.append((obs, state, action, reward, info))
            obs = new_obs
            i += 1
        
        Trials.append(trials)
    AllTrials[delay] = Trials
    rews = [np.sum([trial[-2] for trial in trials]) for trials in Trials]
    mu = np.mean(rews)
    se = np.std(rews)/np.sqrt(len(rews))
    perfs.append((delay, mu, se, rews))

for delay,mu,se,rews in perfs:
    plt.plot(delay*np.ones(len(rews)) + 0.1*np.random.randn(len(rews)), rews, 'k.', markersize=1, zorder=-1, alpha=0.2)
    plt.plot(delay, mu, 'ko')
    plt.plot(delay*np.ones(2), [mu-se,mu+se], 'k-', linewidth=1)
    print('{:0.2f} ± {:0.2f}'.format(mu, se))
plt.xticks(delays)
plt.ylim([0,1.01*env.max_timesteps_per_episode])
plt.xlabel('delay')
plt.ylabel('mean episode duration')

#%% visualize trjaectories

from sklearn.decomposition import PCA
Z = np.vstack([zs[0] for trials in AllTrials[3] for o,zs,a,r,i in trials])
pca = PCA(n_components=Z.shape[1])
pca.fit(Z)
plt.plot(pca.explained_variance_ratio_[:10], '.-')
plt.ylim([0,1])

# for trials in AllTrials[2][:10]:
#     z = np.vstack([zs[0] for o,zs,a,r,i in trials])
#     z = pca.transform(z)
#     h = plt.plot(z[:,0], z[:,1], '-', alpha=0.5, linewidth=1)
#     plt.plot(z[0,0], z[0,1], '+', color=h[0].get_color())

#%% use latent LSTM activity to regress predictive state representations

delays = list(range(10))
useCellState = False
Pts = {delay: ([], [], []) for delay in delays}

Trials = AllTrials[env_config['delay']] # get rollouts using the environment with matching delay
Trials = AllTrials[3] # get rollouts using the environment with matching delay

# train: delay=3
# O[t] = S[t-3]
# Z[t] = f(O[t]) = f(S[t-3]) ≈ S[t] = O[t+3]
# 
# test: exp delay=2, test delay=τ
# X_z[t] := Z[t] = f(O[t]) = f(S[t-2]) ≈ S[t] = O[t+2]
# X_s[t] := S[t] = O[t+2]
# Y[t] := O[t+2] = S[t]

for i, trials in enumerate(Trials):
    X = []
    Y = []
    isTrainTrial = True if i < len(Trials)/2 else False
    for trial in trials:
        obs = trial[0]
        Y.append(obs)

        H,C = trial[1]
        Z = C if useCellState else H
        # Z = trial[-1]['state']
        X.append(Z)

    X = np.vstack(X)
    Y = np.vstack(Y)
    for delay in delays:
        Xc = X if delay == 0 else X[:-delay]
        Yc = Y[delay:]
        # if delay == 2:
        #     assert np.isclose(Xc, Yc).all()
        ixTrainc = np.array([isTrainTrial]*len(Yc))
        Pts[delay][0].append(Xc)
        Pts[delay][1].append(Yc)
        Pts[delay][2].append(ixTrainc)
for delay, (X, Y, I) in Pts.items():
    Pts[delay] = (np.vstack(X), np.vstack(Y), np.hstack(I).astype(bool))

from analysis.correlations import linreg_fit, linreg_eval
def linreg(X, Y, ix):
    mdl = linreg_fit(X[ix], Y[ix], scale=True, add_bias=True)
    res = linreg_eval(X[~ix], Y[~ix], mdl)
    return mdl, res

pts = []
for delay, (X, Y, ixTrain) in Pts.items():
    # use LSTM's activity at time (t-delay) to predict state at time t
    mdl, res = linreg(X, Y, ixTrain)
    pts.append((delay, res['rsq']))
pts = np.vstack(pts)

plt.plot(pts[:,0], pts[:,1], 'ko')
plt.plot(env_config['delay']*np.ones(2), plt.ylim(), 'k--', zorder=-1, linewidth=1, alpha=0.5)
plt.xlabel('delay')
plt.ylabel('$R^2$')
plt.ylim([-0.03,1.03])

# %%
