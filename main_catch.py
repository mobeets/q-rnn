#%% imports

import numpy as np
from plotting.base import plt
import gymnasium as gym
from ray.rllib.algorithms import r2d2
from ray.tune.registry import register_env
from tasks.catch import CatchEnv

#%% view env

render_mode = None
env = CatchEnv(render_mode=render_mode, gravity=0)

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

ends = [X[X[:,0]==i][-1,2] for i in np.unique(X[:,0])]
plt.hist(ends, np.arange(0, env.screen_height,20))

#%% initialize model

env_config = {'gravity': 0}
env_name = 'catch'
register_env(env_name, lambda env_config: CatchEnv(**env_config))

config = r2d2.R2D2Config()
config = config.environment(env_name, env_config=env_config)
# config.rollouts(num_rollout_workers=0)
config.lr = 0.0005
config.gamma = 1
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
config.evaluation_duration = 20; config.evaluation_interval = 1 # evaluate using 10 episodes every episode

algo = config.build()
print(algo.get_policy().model)

#%% train model

outputs = []
best_score = -np.inf
checkpoints = []

get_score = lambda x: x['evaluation']['episode_reward_mean'] if 'evaluation' in x else x['episode_reward_mean']
for i in range(200):
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

#%% rollout

algo.restore(checkpoints[-2])

explore = False
nepisodes = 100

env = CatchEnv(**env_config)

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
        new_obs, reward, terminated, truncated, info = env.step(action)
        trials.append((obs, state, action, reward, info))
        obs = new_obs
        i += 1
    
    Trials.append(trials)

print(np.mean([trials[-1][-2] for trials in Trials]))
