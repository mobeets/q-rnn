#%% imports

import numpy as np
from plotting.base import plt
import gymnasium as gym
from ray.rllib.algorithms import r2d2
from ray.tune.registry import register_env
from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole
from tasks.cartpole import DelayedStatelessCartpole

#%% initialize model

env_config = {'delay': 2}
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
config.evaluation_duration = 10; config.evaluation_interval = 1 # evaluate using 1 episode every episode

algo = config.build()
print(algo.get_policy().model)

#%% train model

outputs = []
best_score = 0
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

print('Best score: {}\n    Checkpoint: '.format(best_score, checkpoints[-1]))
checkpoints.append(algo.save())
# algo.restore(checkpoints[-2]) # restore best scoring model

plt.plot([get_score(x) for x in outputs])

#%% plot results

plt.plot([get_score(x) for x in outputs])

#%% rollout

delays = list(np.arange(6))
nepisodes = 100
perfs = []

for delay in delays:
    env_config = {'delay': delay}
    env = DelayedStatelessCartpole(**env_config)

    Trials = []
    for _ in range(nepisodes):
        terminated = truncated = False
        obs, info = env.reset()

        state = algo.get_policy().get_initial_state()
        action = 0
        reward = 0

        trials = []
        i = 0
        while not terminated and not truncated:
            # action, state, _ = algo.compute_single_action(obs, state=state)
            action, state, _ = algo.compute_single_action(obs, prev_action=action, state=state)
            # action, state, _ = algo.compute_single_action(obs, prev_action=action, prev_reward=reward, state=state)
            obs, reward, terminated, truncated, info = env.step(action)
            trials.append((obs, state, action, reward))
            i += 1

        Trials.append(trials)
        # plt.plot(np.vstack([(trial[0][0], np.rad2deg(trial[0][1])) for trial in trials]))
        # plt.plot(np.vstack([trial[1][0] for trial in trials]))

    rews = [np.sum([trial[-1] for trial in trials]) for trials in Trials]
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

# %%
