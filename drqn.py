# source: https://github.com/keep9oing/DRQN-Pytorch-CartPole-v1
import os.path
import datetime
import json
from types import ModuleType

import numpy as np
import torch
import torch.optim as optim
device = torch.device('cpu')

from model import DRQN
from train import train
from replay import EpisodeMemory, EpisodeBuffer
from plotting import plot_results
from tasks.roitman2002 import Roitman2002
from tasks.beron2022 import Beron2022_TrialLevel

# parameters
learning_rate = 1e-3 # initial learning rate
buffer_len = int(100000)
min_epi_num = 8 # episodes to run before training the Q network
episodes = 1500 # total number of episodes to train on
# ntrials_per_episode = 20 # number of trials comprising an episode
ntrials_per_episode = 800 # number of trials comprising an episode
print_per_iter = 1 # episodes between printing status
target_update_period = 4 # time steps between target network updates
tau = 1e-2 # exponential smoothing parameter for target network update
max_trial_length = 1000 # max number of time steps in episode
eps_start = 0.5 # initial epsilon used in policy
eps_end = 0.001 # final epsilon used in policy
eps_decay = 0.995 # time constant of decay for epsilon used in policy

# training params
batch_size = 8
random_update = True # If you want to do random update instead of sequential update
lookup_step = 100 # number of time steps in sampled episode
max_epi_len = 600 # max number of time steps used in sample episode
max_epi_num = 100 # max number of episodes remembered
gamma = 0.9 # reward discount factor

# other params
include_prev_reward = True
include_prev_action = True

def save_params(environment, run_name, hidden_size, filenames):
    params = dict((x,y) for x,y in globals().items() if not x.startswith('__') and not callable(y) and not isinstance(y, ModuleType))
    params.pop('device')
    params['time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    params['environment'] = environment
    params['run_name'] = run_name
    params['hidden_size'] = hidden_size
    params['filenames'] = filenames
    with open(filenames['paramsfile'], 'w') as f:
        json.dump(params, f)

def save_results(all_trials, outfile):
    np.save(outfile, all_trials)

def train_model(environment, run_name=None, hidden_size=4):
    if run_name is None:
        run_name = str(datetime.datetime.now()).replace(' ', '-').replace('.', '-').replace(':', '-')

    # Set gym environment
    if environment == 'roitman2002':
        env_params = {'reward_amounts': [20, -400, -400, -1]}
        env = Roitman2002(**env_params)
    elif environment == 'beron2022':
        env_params = {'p_rew_max': 0.8, 'ntrials': 1}
        env = Beron2022_TrialLevel(**env_params)

    # prepare to save
    save_dir = 'data/models'
    weightsfile_initial = os.path.join(save_dir, 'weights_initial_h{}_{}.pth'.format(hidden_size, run_name))
    weightsfile_final = os.path.join(save_dir, 'weights_final_h{}_{}.pth'.format(hidden_size, run_name))
    scoresfile = os.path.join(save_dir, 'results_h{}_{}.npy'.format(hidden_size, run_name))
    plotfile = os.path.join(save_dir, 'results_h{}_{}.pdf'.format(hidden_size, run_name))
    paramsfile = os.path.join(save_dir, 'results_h{}_{}.json'.format(hidden_size, run_name))
    filenames = {'weightsfile_initial': weightsfile_initial,
        'weightsfile_final': weightsfile_final,
        'scoresfile': scoresfile,
        'plotfile': plotfile,
        'paramsfile': paramsfile
    }
    save_params(environment, run_name, hidden_size, filenames)

    Q = DRQN(input_size=1 + include_prev_reward + include_prev_action*env.action_space.n, # stim and reward
                hidden_size=hidden_size,
                output_size=env.action_space.n).to(device)
    Q_target = DRQN(input_size=Q.input_size, # stim and reward
                        hidden_size=Q.hidden_size,
                        output_size=Q.output_size).to(device)
    Q_target.load_state_dict(Q.state_dict())
    Q.save_weights_to_path(filenames['weightsfile_initial'], Q.initial_weights)

    # set optimizer
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

    epsilon = eps_start
    episode_memory = EpisodeMemory(random_update=random_update, 
                                    max_epi_num=max_epi_num,
                                    max_epi_len=max_epi_len, 
                                    batch_size=batch_size, 
                                    lookup_step=lookup_step)

    # train
    trials = []
    all_trials = []
    for i in range(episodes):
        r = 0
        a_prev = np.zeros(env.action_space.n)
        h = Q.init_hidden_state(training=False)

        episode_record = EpisodeBuffer()
        for j in range(ntrials_per_episode):
            obs = np.array([env.reset()[0]])
            if include_prev_reward:
                obs = np.hstack([obs, [r]]) # add previous reward
            if include_prev_action:
                obs = np.hstack([obs, a_prev]) # add previous action
            done = False
            
            t = 0
            r_sum = 0
            while not done or t > max_trial_length:
                # get action
                a, (q,h) = Q.sample_action(torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0), 
                                        h.to(device), epsilon)

                # take action
                obs_prime, r, done, truncated, info = env.step(a)
                obs_prime = np.array([obs_prime])
                if include_prev_reward:
                    obs_prime = np.hstack([obs_prime, [r]]) # add previous reward
                if include_prev_action:
                    a_prev = np.zeros(env.action_space.n); a_prev[a] = 1.
                    obs_prime = np.hstack([obs_prime, a_prev]) # add previous action

                # make data
                episode_record.put([obs, a, r/100.0, obs_prime, 0.0 if done else 1.0])
                obs = obs_prime
                r_sum += r

                if len(episode_memory) >= min_epi_num:
                    train(Q, Q_target, episode_memory, device, 
                            optimizer=optimizer,
                            batch_size=batch_size,
                            gamma=gamma)

                    if (t+1) % target_update_period == 0:
                        for target_param, local_param in zip(Q_target.parameters(), Q.parameters()): # <- soft update
                                target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
                    
                t += 1
                if done:
                    if environment == 'roitman2002':
                        trials.append([env.t - env.iti, env.state, a, r_sum, env.t < env.iti, r > 0])
                    elif environment == 'beron2022':
                        trials.append([info['state'], a, r])
                    break
        episode_memory.put(episode_record)
        epsilon = max(eps_end, epsilon * eps_decay) # linear annealing on epsilon

        if i % print_per_iter == 0 and i > 0:
            all_trials.extend(trials)
            ctrials = np.vstack(trials)

            print("nepisode: {}, nbuffer: {}, avgtriallength: {:0.2f}, avgaborts: {:0.2f}, avgrewards: {:0.2f}, nepisodes: {}, 100*eps: {:.1f}%".format(
                i, len(episode_memory), 
                ctrials[:,0].mean(), ctrials[:,-2].mean(), ctrials[:,-1].mean(), len(ctrials), epsilon*100))
            trials = []
            Q.checkpoint_weights()
            Q.save_weights_to_path(filenames['weightsfile_final'], Q.saved_weights)
            save_results(all_trials, filenames['scoresfile'])
            # plot_results(all_trials, ntrials_per_episode, filenames['plotfile'])
    env.close()

    Q.checkpoint_weights()
    Q.save_weights_to_path(filenames['weightsfile_final'], Q.saved_weights)
    save_results(all_trials, filenames['scoresfile'])
    # plot_results(all_trials, ntrials_per_episode, filenames['plotfile'])

def call_main_inner(**args):
    run_index = args.pop('run_index')
    run_name = 'h{}_v{}'.format(args['hidden_size'], run_index)
    print('======= RUN {} ========'.format(run_name))
    train_model(environment, run_name, hidden_size=args['hidden_size'])

def parallelize():
    import multiprocessing
    from multiprocessing.pool import ThreadPool

    # Runs repeats in parallel
    CPU_COUNT = multiprocessing.cpu_count()
    print("Found {} cpus".format(CPU_COUNT))
    pool = ThreadPool(CPU_COUNT)

    for h in [4,8]:
        for i in range(20):
            targs = {}
            targs['hidden_size'] = h
            targs['run_index'] = i
            pool.apply_async(call_main_inner, kwds=targs)

    pool.close()
    pool.join()

if __name__ == '__main__':
    environment = 'beron2022'
    run_name = 'beron_v3_p08'
    train_model(environment, run_name, hidden_size=3)
