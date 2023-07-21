# source: https://github.com/keep9oing/DRQN-Pytorch-CartPole-v1
import os.path
import datetime
import json
from types import ModuleType
import argparse
from argparse import Namespace

import numpy as np
import torch
import torch.optim as optim
device = torch.device('cpu')

from model import DRQN
from train import train, probe_model
from replay import EpisodeMemory, EpisodeBuffer
from tasks.roitman2002 import Roitman2002
from tasks.beron2022 import Beron2022_TrialLevel

# training parameters
learning_rate = 1e-3 # initial learning rate
buffer_len = int(100000)
min_epi_num = 8 # episodes to run before training the Q network
q_tau = 1e-2 # exponential smoothing parameter for target network update

# policy parameters
eps_start = 0.5 # initial epsilon used in policy
eps_end = 0.001 # final epsilon used in policy
# eps_start = 2; eps_end = 1; print("WARNING: eps always 1")
eps_decay = 0.995 # time constant of decay for epsilon used in policy
tau_start = 2.0 # initial tau used in softmax policy
tau_end = 0.001 # final tau used in softmax policy
tau_decay = 0.995 # time constant of decay for tau used in softmax policy

# memory params
random_update = True # If you want to do random update instead of sequential update
lookup_step = 100 # number of time steps in sampled episode
max_epi_len = 600 # max number of time steps used in sample episode
max_epi_num = 100 # max number of episodes remembered

def save_params(args, filenames, scores=None):
    params = dict((x,y) for x,y in globals().items() if not x.startswith('__') and not callable(y) and not isinstance(y, ModuleType))
    params.pop('device')
    params.pop('parser')
    params.pop('args')
    params.update(vars(args))
    params['time_last_updated'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    params['filenames'] = filenames
    if scores is not None:
        params['scores'] = scores
    with open(filenames['paramsfile'], 'w') as f:
        json.dump(params, f)

def get_filenames(args):
    weightsfile_initial = os.path.join(args.save_dir, 'weights_initial_h{}_{}.pth'.format(args.hidden_size, args.run_name))
    weightsfile_final = os.path.join(args.save_dir, 'weights_final_h{}_{}.pth'.format(args.hidden_size, args.run_name))
    paramsfile = os.path.join(args.save_dir, 'results_h{}_{}.json'.format(args.hidden_size, args.run_name))
    filenames = {
        'weightsfile_initial': weightsfile_initial,
        'weightsfile_final': weightsfile_final,
        'paramsfile': paramsfile
    }
    return filenames

def balance_model_to_explore_actions(model, env, args):
    # run on an episode, then adjust the mean Q value so that actions are equally likely
    model.output.bias = torch.nn.Parameter(torch.zeros(2))
    trials = probe_model(model, env, 1, args.ntrials_per_episode,
                        epsilon=1, # random policy
                        include_prev_reward=args.include_prev_reward,
                        include_prev_action=args.include_prev_action)
    Q = np.vstack([x.Q for x in trials])
    model.output.bias = torch.nn.Parameter(torch.Tensor(-Q.mean(axis=0)))

def train_model(args):
    # set gym environment
    if args.experiment == 'roitman2002':
        env_params = {'reward_amounts': [20, -400, -400, -1]}
        env = Roitman2002(**env_params)
    elif args.experiment == 'beron2022':
        env_params = {'p_rew_max': args.p_reward_max}
        env = Beron2022_TrialLevel(**env_params)

    # create models
    Q = DRQN(input_size=1 + args.include_prev_reward + args.include_prev_action*env.action_space.n, # stim and reward
                hidden_size=args.hidden_size,
                output_size=env.action_space.n).to(device)
    balance_model_to_explore_actions(Q, env, args)
    Q_target = DRQN(input_size=Q.input_size, # stim and reward
                        hidden_size=Q.hidden_size,
                        output_size=Q.output_size).to(device)
    Q_target.load_state_dict(Q.state_dict())
    
    # prepare to save
    filenames = get_filenames(args)
    save_params(args, filenames)
    Q.save_weights_to_path(filenames['weightsfile_initial'], Q.initial_weights)

    # set optimizer
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

    if args.use_softmax_policy:
        epsilon = None
        tau = tau_start
    else:
        epsilon = eps_start
        tau = None
    episode_memory = EpisodeMemory(random_update=random_update, 
                                    max_epi_num=max_epi_num,
                                    max_epi_len=max_epi_len, 
                                    batch_size=args.batch_size, 
                                    lookup_step=lookup_step)

    # train
    best_score = -np.inf
    scores = []
    for i in range(args.episodes):
        r = 0
        a_prev = np.zeros(env.action_space.n)
        h = Q.init_hidden_state(training=False)
        cur_loss = 0
        t = 0

        episode_record = EpisodeBuffer()
        for _ in range(args.ntrials_per_episode):
            obs = np.array([env.reset()[0]])
            if args.include_prev_reward:
                obs = np.hstack([obs, [r]]) # add previous reward
            if args.include_prev_action:
                obs = np.hstack([obs, a_prev]) # add previous action
            done = False
            
            while not done:
                # get action
                a, (q,h) = Q.sample_action(torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0), 
                                        h.to(device), epsilon=epsilon, tau=tau)

                # take action 
                obs_next, r, done, truncated, info = env.step(a)
                obs_next = np.array([obs_next])
                if args.include_prev_reward:
                    obs_next = np.hstack([obs_next, [r]]) # add previous reward
                if args.include_prev_action:
                    a_prev = np.zeros(env.action_space.n); a_prev[a] = 1.
                    obs_next = np.hstack([obs_next, a_prev]) # add previous action

                # make data
                episode_record.put([obs, a, r/100.0, obs_next, 0.0 if done else 1.0])
                obs = obs_next

                if len(episode_memory) >= min_epi_num:
                    # update Q network
                    cur_loss += train(Q, Q_target, episode_memory, device, 
                            optimizer=optimizer,
                            batch_size=args.batch_size,
                            gamma=args.gamma,
                            l2_penalty=args.l2_penalty)

                    if (t+1) % args.target_update_period == 0:
                        # update Q_target network
                        for target_param, local_param in zip(Q_target.parameters(), Q.parameters()): # <- soft update
                            target_param.data.copy_(q_tau*local_param.data + (1.0 - q_tau)*target_param.data)
                t += 1
        episode_memory.put(episode_record)
        if args.use_softmax_policy:
            tau = max(tau_end, tau * tau_decay) # linear annealing on tau
        else:
            epsilon = max(eps_end, epsilon * eps_decay) # linear annealing on epsilon

        # evaluate model using greedy policy
        test_trials = probe_model(Q, env, 1, args.ntrials_per_episode,
                             epsilon=0, # always evaluate with greedy policy
                             include_prev_reward=args.include_prev_reward,
                             include_prev_action=args.include_prev_action)
        cur_score = np.hstack([x.R for x in test_trials]).mean()
        scores.append(cur_score)

        if cur_score > best_score:
            best_score = cur_score
            print("New top score: {:0.3f}".format(best_score))
            Q.checkpoint_weights()
            Q.save_weights_to_path(filenames['weightsfile_final'], Q.saved_weights)
            save_params(args, filenames, scores)

        if i % args.print_per_iter == 0 and i > 0:
            print("episode {} | loss: {:0.4f}, score: {:0.3f}, {}={:0.1f}%".format(i, cur_loss, cur_score,
                            'τ' if args.use_softmax_policy else 'ε',
                            100*(tau-tau_start)/(tau_end-tau_start) if args.use_softmax_policy else 100*(epsilon-eps_start)/(eps_end-eps_start)))
    env.close()

def train_model_outer(**args):
    run_index = args.pop('run_index')
    if type(args) is dict:
        args = Namespace(**args)
    print('======= RUN {} ========'.format(run_index))
    train_model(args)

def parallel_train():
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
            pool.apply_async(train_model_outer, kwds=targs)

    pool.close()
    pool.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, help='tag for current run',
                        default=str(datetime.datetime.now()).replace(' ', '-').replace('.', '-').replace(':', '-'))
    parser.add_argument('-d', '--save_dir', type=str,
                        default='data/models', help='where to save trained model')
    parser.add_argument('--experiment', type=str,
                        default='beron2022', choices=['beron2022', 'roitman2002'])
    parser.add_argument('--p_reward_max', type=float,
                        default=0.8, help='max reward probability of arm (beron2022 env only)')
    parser.add_argument('-k', '--hidden_size', type=int,
                        default=10, help='number of hidden units in the rnn')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=8, help='batch size used during training')
    parser.add_argument('-e', '--episodes', type=int,
                        default=200, help='total number of episodes to train on')
    parser.add_argument('--ntrials_per_episode', type=int,
                        default=800, help='number of trials comprising an episode')
    parser.add_argument('--print_per_iter', type=int,
                        default=1, help='episodes between printing status')
    parser.add_argument('--target_update_period', type=int,
                        default=1, help='time steps between target network updates')
    parser.add_argument('-g', '--gamma', type=float,
                        default=0.9, help='reward discount factor')
    parser.add_argument('--l2_penalty', type=float,
                        default=0, help='penalty on L2 norm of RNN activations')
    parser.add_argument('--use_softmax_policy', action='store_true',
                        default=False, help='if False, uses epsilon-greedy policy')
    parser.add_argument('--include_prev_reward', action='store_true',
                        default=False)
    parser.add_argument('--include_prev_action', action='store_true',
                        default=False)
    args = parser.parse_args()
    if args.experiment == 'beron2022':
        print('WARNING: For {}, auto-including prev reward and action.'.format(args.experiment))
        args.include_prev_reward = True
        args.include_prev_action = True
    print(args)
    train_model(args)
