# source: https://github.com/keep9oing/DRQN-Pytorch-CartPole-v1
import os.path
import numpy as np
import torch
import torch.optim as optim
device = torch.device('cpu')

from model import DRQN
from train import train
from replay import EpisodeMemory, EpisodeBuffer
from plotting import plot_results
from tasks.roitman2002 import Roitman2002

# Set gym environment
reward_amounts = [20, -400, -1]
env = Roitman2002(reward_amounts=reward_amounts)

# Set parameters
learning_rate = 1e-3 # initial learning rate
buffer_len = int(100000)
min_epi_num = 8 # episodes to run before training the Q network
episodes = 800 # total number of episodes to train on
ntrials_per_episode = 20 # number of trials comprising an episode
print_per_iter = 25 # episodes between printing status
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

# create model
hidden_size = 4
Q = DRQN(input_size=2, # stim and reward
            hidden_size=hidden_size,
            output_size=env.action_space.n).to(device)
Q_target = DRQN(input_size=2, # stim and reward
                    hidden_size=hidden_size,
                    output_size=env.action_space.n).to(device)
Q_target.load_state_dict(Q.state_dict())

# prepare to save
save_dir = 'data/drqn'
weightsfile_initial = os.path.join(save_dir, 'weights_initial.pth')
weightsfile_final = os.path.join(save_dir, 'weights_final.pth')
plotfile = os.path.join(save_dir, 'results.pdf')

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
    h = Q.init_hidden_state(batch_size=batch_size, training=False)

    episode_record = EpisodeBuffer()
    for j in range(ntrials_per_episode):
        obs = env.reset()[0]
        obs = np.array([obs, r]) # add previous reward
        done = False
        
        t = 0
        r_sum = 0
        while not done or t > max_trial_length:
            # get action
            a, h = Q.sample_action(torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0), 
                                    h.to(device), epsilon)

            # take action
            obs_prime, r, done, truncated, info = env.step(a)
            obs_prime = np.array([obs_prime, r]) # add previous reward

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
                trials.append([env.t - env.iti, env.state, a, r_sum, env.t < env.iti, r > 0])
                break
    episode_memory.put(episode_record)
    epsilon = max(eps_end, epsilon * eps_decay) # linear annealing on epsilon

    if i % print_per_iter == 0 and i > 0:
        all_trials.extend(trials)
        ctrials = np.vstack(trials)

        print("nepisode: {}, nbuffer: {}, avgtriallength: {}, naborts: {}, nrewards: {}, nepisodes: {}, 100*eps: {:.1f}%".format(
            i, len(episode_memory), 
            ctrials[:,0].mean(), ctrials[:,-2].sum(), ctrials[:,-1].sum(), len(ctrials), epsilon*100))
        trials = []
        Q.checkpoint_weights()
        Q.save_weights_to_path(weightsfile_initial, Q.initial_weights)
        Q.save_weights_to_path(weightsfile_final, Q.saved_weights)
        plot_results(all_trials, ntrials_per_episode, plotfile)
env.close()

Q.checkpoint_weights()
Q.save_weights_to_path(weightsfile_initial, Q.initial_weights)
Q.save_weights_to_path(weightsfile_final, Q.saved_weights)
plot_results(all_trials, ntrials_per_episode, plotfile)
