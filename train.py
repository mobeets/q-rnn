import numpy as np
import torch
import torch.nn.functional as F
from tasks.trial import Trial

device = torch.device('cpu')

def train(q_net=None, target_q_net=None, episode_memory=None,
          device=None, 
          optimizer=None,
          batch_size=1,
          gamma=0.99):

    assert device is not None, "None Device input: device should be selected."

    # Get batch from replay buffer
    samples, seq_len = episode_memory.sample()

    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    for i in range(batch_size):
        observations.append(samples[i]["obs"])
        actions.append(samples[i]["acts"])
        rewards.append(samples[i]["rews"])
        next_observations.append(samples[i]["next_obs"])
        dones.append(samples[i]["done"])

    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_observations = np.array(next_observations)
    dones = np.array(dones)

    observations = torch.FloatTensor(observations.reshape(batch_size,seq_len,-1)).to(device)
    actions = torch.LongTensor(actions.reshape(batch_size,seq_len,-1)).to(device)
    rewards = torch.FloatTensor(rewards.reshape(batch_size,seq_len,-1)).to(device)
    next_observations = torch.FloatTensor(next_observations.reshape(batch_size,seq_len,-1)).to(device)
    dones = torch.FloatTensor(dones.reshape(batch_size,seq_len,-1)).to(device)

    with torch.no_grad():
        h_target = target_q_net.init_hidden_state(batch_size=batch_size, training=True)
        q_target, _ = target_q_net(next_observations, h_target.to(device))
        q_target_max = q_target.max(2)[0].view(batch_size,seq_len,-1).detach()
        targets = rewards + gamma*q_target_max*dones

    h = q_net.init_hidden_state(batch_size=batch_size, training=True)
    q_out, _ = q_net(observations, h.to(device))
    q_a = q_out.gather(2, actions)

    # Multiply Importance Sampling weights to loss        
    loss = F.smooth_l1_loss(q_a, targets)
    
    # Update Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def probe_model(model, env, nepisodes, ntrials_per_episode, epsilon=0):
    trials = []
    with torch.no_grad():
        for i in range(nepisodes):
            h = model.init_hidden_state(training=False)
            r = 0

            for j in range(ntrials_per_episode):
                obs_next = env.reset()[0]
                done = False
                
                trial = Trial(env.state, env.iti, index_in_episode=j)
                while not done:
                    # get action
                    obs = np.array([obs_next, r]) # add previous reward
                    a, (q, h) = model.sample_action(torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0), 
                                            h.to(device), epsilon=epsilon)

                    # take action
                    obs_next, r, done, truncated, info = env.step(a)

                    # save
                    trial.update(obs, a, r, h.numpy(), q.numpy())
                trials.append(trial)
    return trials

def probe_model_off_policy(model, policymodel, env, nepisodes, ntrials_per_episode, epsilon=0):
    trials = []
    with torch.no_grad():
        for i in range(nepisodes):
            h = model.init_hidden_state(training=False)
            hp = policymodel.init_hidden_state(training=False)
            r = 0

            for j in range(ntrials_per_episode):
                obs_next = env.reset()[0]
                done = False
                
                trial = Trial(env.state, env.iti, index_in_episode=j)
                while not done:
                    # get action
                    obs = np.array([obs_next, r]) # add previous reward
                    cobs = torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0)
                    _, (q, h) = model.sample_action(cobs, h.to(device), epsilon=epsilon)
                    a, (_, hp) = policymodel.sample_action(cobs, hp.to(device), epsilon=epsilon)

                    # take action
                    obs_next, r, done, truncated, info = env.step(a)

                    # save
                    trial.update(obs, a, r, h.numpy(), q.numpy())
                trials.append(trial)
    return trials
