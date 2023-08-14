#%% imports

import gymnasium as gym
from ray.rllib.algorithms import r2d2
from ray.tune.registry import register_env
from tasks.beron2022 import Beron2022, Beron2022_TrialLevel
import matplotlib.pyplot as plt

#%% initialize model

env_config = {'p_rew_max': 0.8, 'p_switch': 0.02, 'ntrials': 800}
register_env('beron2022', lambda env_config: Beron2022_TrialLevel(**env_config))

config = r2d2.R2D2Config()
config = config.environment('beron2022', env_config=env_config)
config.lr = 1e-3
config.gamma = 0.0
config.exploration_config['initial_epsilon'] = 0.5
config.exploration_config['final_epsilon'] = 0.2
config.exploration_config['epsilon_timesteps'] = 30000
config.dueling = False
config.hiddens = []
config.model['use_lstm'] = True
config.model['lstm_cell_size'] = 10
config.model['lstm_use_prev_action'] = True
config.model['lstm_use_prev_reward'] = True
config.model['max_seq_len'] = 20
config.model['fcnet_hiddens'] = []
config.model['fcnet_activation'] = 'linear'
config.evaluation_duration = 1; config.evaluation_interval = 1 # evaluate using 1 episode every episode

algo = config.build()
print(algo.get_policy().model)

#%% train model
# best: 480

outputs = []
best_score = 0
for i in range(200):
    output = algo.train()
    outputs.append(output)

    score = outputs[-1]['episode_reward_mean']
    epsilons = algo.workers.foreach_worker(lambda worker: worker.get_policy().exploration.get_state()['cur_epsilon'])
    print(i, score, epsilons)
    if score > best_score:
        best_score = score
        checkpoint_dir = algo.save()
        print("Saved checkpoint to {}.".format(checkpoint_dir))

# algo.restore(checkpoint_dir)
# checkpoint_dir = '/Users/mobeets/ray_results/R2D2_ray.rllib.examples.env.stateless_cartpole.StatelessCartPole_2023-08-14_10-46-412o5u0u1a/checkpoint_000126'

#%% plot results

plt.plot([x['episode_reward_mean'] for x in outputs])

#%% rollout

# env = Beron2022_TrialLevel(**env_config)
from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole
env = StatelessCartPole()

nepisodes = 20
Trials = []
for _ in range(nepisodes):
    terminated = truncated = False
    obs, info = env.reset()

    state = algo.get_policy().get_initial_state()
    action = 0
    reward = 0

    trials = []
    while not terminated and not truncated:
        action, state, _ = algo.compute_single_action(obs, state=state)
        # action, state, _ = algo.compute_single_action(obs, prev_action=action, prev_reward=reward, state=state)
        obs, reward, terminated, truncated, info = env.step(action)
        trials.append((obs, state, action, reward))

    Trials.append(trials)
    plt.plot(np.vstack([(trial[0][0], np.rad2deg(trial[0][1])) for trial in trials]))
    # plt.plot(np.vstack([trial[1][0] for trial in trials]))

print(np.mean([np.sum([trial[-1] for trial in trials]) for trials in Trials]))

#%% initialize model

env_config = {}
# register_env('beron2022', lambda env_config: Beron2022_TrialLevel(**env_config))

env_name = 'ray.rllib.examples.env.stateless_cartpole.StatelessCartPole'
config = r2d2.R2D2Config()
config = config.environment(env_name, env_config=env_config)
# config.rollouts(num_rollout_workers=0)
config.lr = 0.0005
# config.gamma = 0.9 # 0.997
# config.exploration_config['initial_epsilon'] = 0.5
# config.exploration_config['final_epsilon'] = 0.2
config.exploration_config['epsilon_timesteps'] = 50000 # divide by 1000 to get number of batches
# config.dueling = False
# config.hiddens = []
config.model['use_lstm'] = True
config.model['lstm_cell_size'] = 64
# config.model['lstm_use_prev_action'] = True
# config.model['lstm_use_prev_reward'] = True
config.model['max_seq_len'] = 20
config.model['fcnet_hiddens'] = [64]
config.model['fcnet_activation'] = 'linear'
config.replay_buffer_config['replay_burn_in'] = 20
# config.evaluation_duration = 1; config.evaluation_interval = 1 # evaluate using 1 episode every episode

algo = config.build()
print(algo.get_policy().model)

# %%
