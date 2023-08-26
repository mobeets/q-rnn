#%% imports

import numpy as np
from plotting.base import plt
import gymnasium as gym
from ray.rllib.algorithms.pg import PGConfig
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.algorithms.a2c import A2CConfig
from ray.tune.registry import register_env
from tasks.centerout import CenterOutEnv

#%% view env

render_mode = None
env = CenterOutEnv(tau=20, render_mode=render_mode)

nepisodes = 10
Trials = []
for j in range(nepisodes):
    obs, info = env.reset()
    terminated = False
    truncated = False
    i = 0
    trials = []
    while not (terminated or truncated):
        action = env.action_space.sample()
        trg = obs[:2]; pos = obs[2:]; action = (trg-pos)
        obs, reward, terminated, truncated, info = env.step(action)
        trials.append((i, j, obs[:2], obs[2:], action, reward))
        i += 1
    Trials.append(trials)
if render_mode is not None:
    env.close()
print(np.mean([trials[-1][-1] for trials in Trials]))
print(np.mean([len(trials) for trials in Trials]))

for trials in Trials:
    pts = np.vstack([trial[3] for trial in trials])
    plt.plot(pts[:,0], pts[:,1], '.-')
    pts = np.vstack([trial[2] for trial in trials])
    plt.plot(pts[:,0], pts[:,1], 'o')
plt.axis('equal'), plt.axis('off')

#%% initialize model

env_config = {'progress_weight': 1}
env_name = 'centerout'
register_env(env_name, lambda env_config: CenterOutEnv(**env_config))

config = PGConfig()
# config = ImpalaConfig()
# config = A2CConfig()

config = config.environment(env_name, env_config=env_config)
config.lr = 0.0005
config.gamma = 1

config.model['use_lstm'] = True
config.model['lstm_cell_size'] = 64
config.model['lstm_use_prev_action'] = False
config.model['max_seq_len'] = 101
config.model['fcnet_hiddens'] = [64]
config.model['fcnet_activation'] = 'linear'

algo = config.build()
print(algo.get_policy().model)

#%% train model

# outputs = []; best_score = -np.inf; checkpoints = []

get_score = lambda x: x['evaluation']['episode_reward_mean'] if 'evaluation' in x else x['episode_reward_mean']
for i in range(500):
    output = algo.train()
    outputs.append(output)

    score = get_score(outputs[-1])
    print('{}. Score={:0.3f}, Best={:0.3f}'.format(i, score, best_score))

    if score > best_score:
        # note: this isn't always the best model...
        best_score = score
        checkpoint_dir = algo.save()
        checkpoints.append(checkpoint_dir)
        print("Saved checkpoint to {}.".format(checkpoint_dir))

checkpoints.append(algo.save())
print('Best score: {}\n    Checkpoint: {}'.format(best_score, checkpoints[-2]))
# algo.restore(checkpoints[-2]) # restore best scoring model

plt.plot([get_score(x) for x in outputs])

#%% roll-out

algo.restore(checkpoints[-2])

nepisodes = 64
randomPolicy = False
env = CenterOutEnv(**env_config)

Trials = []
outcomes = []
for j in range(nepisodes):
    terminated = truncated = False
    theta = env.target_angles[j % len(env.target_angles)]
    obs, info = env.reset(options={'theta': theta})

    state = algo.get_policy().get_initial_state()
    action = np.zeros(env.action_space.shape)

    trials = []
    i = 0
    while not terminated and not truncated:
        outs = algo.compute_single_action(obs,
                prev_action=action if config.model['lstm_use_prev_action'] else None,
                state=state)
        action, state, q_info = outs if config.model['use_lstm'] else (outs, None, None)
        if randomPolicy:
            action = env.action_space.sample()
        new_obs, reward, terminated, truncated, info = env.step(action)
        trials.append((obs, state, action, reward, info))
        obs = new_obs
        i += 1

    outcome = 1 if reward > 0 else (-1 if truncated else 0)
    outcomes.append(outcome)
    Trials.append(trials)

print(np.mean([trials[-1][-2] for trials in Trials]))
print([(nm, np.mean([outcome==o for outcome in outcomes])) for nm,o in {'trunc': -1, 'oob': 0, 'success': 1}.items()])

#%% visualize trajectories

clrs = {}
for trg in env.target_angles:
    trgpos = env._normalize_pos(env._target_coords(trg))
    h = plt.plot(trgpos[0], trgpos[1], 'o', markersize=8)
    clrs[trg] = h[0].get_color()

for trials in Trials:
    clr = clrs[trials[0][-1]['theta']]
    pos = np.vstack([trial[0][2:] for trial in trials])
    plt.plot(pos[:,0], pos[:,1], '.-', color=clr, alpha=0.25, markersize=2)

#%% pca

from sklearn.decomposition import PCA

Z = np.vstack([trial[1][0] for trials in Trials for trial in trials])
pca = PCA(n_components=Z.shape[1])
pca.fit(Z)
# plt.plot(pca.explained_variance_ratio_[:10], '.-'), plt.ylim([0,1])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for trials in Trials:
    zs = np.vstack([trial[1][0] for trial in trials])
    zs = pca.transform(zs)
    clr = clrs[trials[0][-1]['theta']]
    plt.plot(zs[:,0], zs[:,1], zs[:,2], '.-', color=clr, alpha=0.25, markersize=2)
    plt.plot(zs[0,0], zs[0,1], zs[0,2], '+', color=clr, alpha=0.25, markersize=5)

# ax.view_init(azim=-110, elev=7.)
plt.tight_layout()
