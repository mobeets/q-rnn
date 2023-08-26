#%% imports

import numpy as np
from plotting.base import plt
import gymnasium as gym
from ray.rllib.algorithms.r2d2 import R2D2Config
from ray.tune.registry import register_env
from tasks.catch import DelayedCatchEnv

#%% view env

render_mode = None
env = DelayedCatchEnv(render_mode=render_mode, gravity=0, delay=0)

nepisodes = 2000
X = []
for j in range(nepisodes):
    obs, info = env.reset()
    terminated = False
    truncated = False
    i = 0
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        i += 1
        x = env.last_state['ball'].copy()
        X.append(np.hstack([j, x]))
if render_mode is not None:
    env.close()
X = np.vstack(X)

starts = [X[X[:,0]==i][0,2] for i in np.unique(X[:,0])]
ends = [X[X[:,0]==i][-1,2] for i in np.unique(X[:,0])]
plt.hist(starts, np.arange(0, env.screen_height,20), alpha=0.4)
plt.hist(ends, np.arange(0, env.screen_height,20), alpha=0.4)

#%% fit prediction model (using different lags)

from analysis.correlations import linreg_fit, linreg_eval

nepisodes = 2000
X = []
for j in range(nepisodes):
    obs, info = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs = np.nan * obs
        caction = np.zeros(env.action_space.n); caction[action] = 1.
        X.append((obs, caction))
X = np.vstack([np.hstack(x) for x in X])

# fit prediction model (using different lags)
lags = range(1,6)
mdls = {}
for lag in lags:
    Xc = []
    inds = []
    for clag in range(1,lag+2):
        t1 = clag-1
        t2 = len(X)-(lag-clag)-1
        Xcc = X[t1:t2].copy()
        Xc.append(Xcc)
        inds.append((t1,t2,len(X)))
    Yc = Xc[-1][:,:-env.action_space.n] # predict next observation
    Xc = np.hstack(Xc[:-1]) # using prev observations/actions at all previous lags
    
    ixc = np.any(np.isnan(Xc), axis=1) | np.any(np.isnan(Yc), axis=1)
    Xc = Xc[~ixc]; Yc = Yc[~ixc]
    ix = np.random.rand(len(Xc)) < 0.5 # train/test split
    mdl = linreg_fit(Xc[ix], Yc[ix], scale=True, add_bias=True)
    res = linreg_eval(Xc[~ix], Yc[~ix], mdl)
    mdls[lag] = (mdl, res)

print([(lag, res['rsq']) for lag, (mdl, res) in mdls.items()])

#%% initialize model

use_r2d2 = False
use_custom_model = False
env_config = {'gravity': 0, 'tau': 0.015, 'action_penalty': 0.0005, 'delay': 10}

env_name = 'catch'
register_env(env_name, lambda env_config: DelayedCatchEnv(**env_config))

config = R2D2Config()
config = config.environment(env_name, env_config=env_config)
config.lr = 0.0005
config.gamma = 1
config.exploration_config['initial_epsilon'] = 1.0
config.exploration_config['final_epsilon'] = 0.0
config.exploration_config['epsilon_timesteps'] = 50000 # divide by 1000 to get number of batches

if use_custom_model:
    from ray.rllib.models import ModelCatalog
    from rnn import CatchRNNModel
    ModelCatalog.register_custom_model("my_rnn_model", CatchRNNModel)

    config.training(model={
        'custom_model': 'my_rnn_model',
        'custom_model_config': {}
        })
else:
    config.model['use_lstm'] = True
    config.model['lstm_cell_size'] = 64
    config.model['lstm_use_prev_action'] = True
    config.model['max_seq_len'] = 50
    config.model['fcnet_hiddens'] = [64]
    config.model['fcnet_activation'] = 'linear'

config.replay_buffer_config['replay_burn_in'] = 20
config.evaluation_duration = 20; config.evaluation_interval = 1 # evaluate using 10 episodes every episode

algo = config.build()
print(algo.get_policy().model)

#%% train model

outputs = []
best_score = -np.inf
checkpoints = []

# algo.restore(checkpoint_delay7)

get_score = lambda x: x['evaluation']['episode_reward_mean'] if 'evaluation' in x else x['episode_reward_mean']
for i in range(250):
    output = algo.train()
    outputs.append(output)

    score = get_score(outputs[-1])
    epsilons = np.min(algo.workers.foreach_worker(lambda worker: worker.get_policy().exploration.get_state()['cur_epsilon']))
    print('{}. Score={:0.3f}, Best={:0.3f}, ε={:0.3f}'.format(i, score, best_score, epsilons))

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

#%% rollout

# algo.restore(checkpoints[-2])
algo.restore(checkpoint_delay0)
# algo.restore(checkpoint_delay5)
# algo.restore(checkpoint_delay3)
# algo.restore(checkpoint_delay7)
# algo.restore(checkpoint_delay10v2)
# algo.restore(checkpoint_delay9)

explore = False
nepisodes = 250
randomPolicy = True

env = DelayedCatchEnv(**env_config)
env.delay = 9

Trials = []
for j in range(nepisodes):
    terminated = truncated = False
    obs, info = env.reset()

    state = algo.get_policy().get_initial_state()
    action = 0

    trials = []
    i = 0
    while not terminated and not truncated:
        action, state, q_info = algo.compute_single_action(obs,
                prev_action=action if config.model['lstm_use_prev_action'] else None,
                state=state, explore=explore)
        if randomPolicy:
            action = env.action_space.sample()
        new_obs, reward, terminated, truncated, info = env.step(action)
        trials.append((obs, state, action, reward, info))
        obs = new_obs
        i += 1
    
    Trials.append(trials)

print(np.mean([trials[-1][-2] for trials in Trials]))

#%% visualize ball and hand paths

for i, trials in enumerate(Trials):
    ball = np.vstack([trial[-1]['ball'][1] for trial in trials])
    hand = np.vstack([trial[-1]['hand'][1] for trial in trials])
    if i == 9:
        plt.plot(ball)
        plt.plot(hand)
        [plt.plot(x*np.ones(2), y + np.array([-1,1])*env.hand_length/2, 'k-', alpha=0.1, linewidth=1, zorder=-1) for x,y in enumerate(hand)]
        break
plt.ylim([0, env.screen_height])

# todos:
# - use evaluation made of fixed trajectories?
# - what are the signatures of prediction?
# 

#%% decoding future ball states from lstm

from analysis.correlations import linreg_fit, linreg_eval
from sklearn.decomposition import PCA

delays = list(np.arange(-5, 15))
Pts = {delay: ([], [], []) for delay in delays}

Z = np.vstack([trial[1][0] for trials in Trials[:int(len(Trials)/2)] for trial in trials])
pca = PCA(n_components=Z.shape[1])
pca.fit(Z)
# plt.plot(pca.explained_variance_ratio_[:10], '.-'), plt.ylim([0,1])

for i, trials in enumerate(Trials):
    isTrainTrial = True if i < len(Trials)/2 else False
    # ball = np.vstack([trial[-1]['ball'] for trial in trials])
    ball = np.vstack([trial[0][:2] for trial in trials])

    hand = np.vstack([trial[-1]['hand'] for trial in trials])[:,1:]
    lstm = np.vstack([trial[1][1] for trial in trials])

    X = lstm; Y = ball
    
    for delay in delays:
        if delay == 0:
            Xc = X
            Yc = Y
        elif delay > 0:
            Xc = X[:-delay]
            Yc = Y[delay:]
        elif delay < 0:
            Xc = X[-delay:]
            Yc = Y[:delay]
        ixTrainc = np.array([isTrainTrial]*len(Yc))
        Pts[delay][0].append(Xc)
        Pts[delay][1].append(Yc)
        Pts[delay][2].append(ixTrainc)
for delay, (X, Y, I) in Pts.items():
    Pts[delay] = (np.vstack(X), np.vstack(Y), np.hstack(I).astype(bool))

def linreg(X, Y, ix):
    mdl = linreg_fit(X[ix], Y[ix], scale=True, add_bias=True)
    res = linreg_eval(X[~ix], Y[~ix], mdl)
    return mdl, res

pts = []
for delay, (X, Y, ixTrain) in Pts.items():
    # use LSTM's activity at time (t-delay) to predict state at time t
    X = pca.transform(X)[:,:5]
    mdl, res = linreg(X, Y, ixTrain)
    pts.append((delay, res['rsq']))
pts = np.vstack(pts)

plt.plot(pts[:,0], pts[:,1], 'ko')
plt.xlabel('delay')
plt.ylabel('$R^2$')
plt.ylim([-0.03,1.03])

# %%

# plt.plot(pts_delay0[:,0], pts_delay0[:,1], 'o', label='τ=0')
# plt.plot(pts_delay5[:,0], pts_delay5[:,1], 'o', label='τ=5')
# plt.plot(pts[:,0], pts[:,1], 'o', label='τ=5')
# plt.plot(pts_delay0_rand[:,0], pts_delay0_rand[:,1], 'o', label='τ=0')
# plt.plot(pts_delay5_rand[:,0], pts_delay5_rand[:,1], 'o', label='τ=5')
# plt.plot(pts_delay3[:,0], pts_delay3[:,1], 'o', label='τ=3')
# plt.plot(pts3_delay0[:,0], pts3_delay0[:,1], 'o', label='τ=0')
# plt.plot(pts3_delay3[:,0], pts3_delay3[:,1], 'o', label='τ=3')
# plt.plot(pts3_delay5[:,0], pts3_delay5[:,1], 'o', label='τ=5')
# plt.plot(pts_rand[:,0], pts_rand[:,1], 'k-', label='random', zorder=-1)
# plt.plot(pts10_delay10[:,0], pts10_delay10[:,1], 'o', label='τ=10')
# plt.plot(pts10_delayrand[:,0], pts10_delayrand[:,1], 'k-', label='random', zorder=-1)
plt.plot(pts9_delay0[:,0], pts9_delay0[:,1], 'o', label='τ=0')
plt.plot(pts9_delay9[:,0], pts9_delay9[:,1], 'o', label='τ=9')
plt.plot(pts9_delayrand[:,0], pts9_delayrand[:,1], 'k-', label='random', zorder=-1)
plt.xlabel('delay (d)')
plt.ylabel('$R^2$ using $Z_t$ to predict $O_{t+d}$')
plt.ylim([-0.03,1.03])
plt.legend(fontsize=8)
