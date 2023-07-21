import numpy as np
import torch
import torch.nn.functional as F
from tasks.trial import Trial

device = torch.device('cpu')
tol = np.finfo('float').min

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
    
    return loss/batch_size

def probe_model(model, env, nepisodes, ntrials_per_episode, behavior_policy=None,
                epsilon=0, tau=tol, include_prev_reward=True, include_prev_action=True):
    
    trials = []
    with torch.no_grad():
        for i in range(nepisodes):
            h = model.init_hidden_state(training=False)
            if behavior_policy is not None:
                hp = behavior_policy.init_hidden_state(training=False)
            r = 0
            a_prev = np.zeros(env.action_space.n)

            for j in range(ntrials_per_episode):
                obs = np.array([env.reset()[0]])
                if include_prev_reward:
                    obs = np.hstack([obs, [r]]) # add previous reward
                if include_prev_action:
                    obs = np.hstack([obs, a_prev]) # add previous action
                done = False
                
                if hasattr(env, 'iti'):
                    trial = Trial(env.state, env.iti, index_in_episode=j, episode_index=i)
                else:
                    trial = Trial(state=None, index_in_episode=j, episode_index=i)
                while not done:                    
                    # get action
                    cobs = torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0)
                    if behavior_policy is None:
                        a, (q, h) = model.sample_action(cobs, 
                                            h.to(device), epsilon=epsilon, tau=tau)
                    else:
                        _, (q, h) = model.sample_action(cobs, h.to(device), epsilon=epsilon, tau=tau)
                        a, (_, hp) = behavior_policy.sample_action(cobs, hp.to(device), epsilon=epsilon, tau=tau)

                    # take action
                    obs_next, r, done, truncated, info = env.step(a)

                    # build next obs
                    obs_next = np.array([obs_next])
                    if include_prev_reward:
                        obs_next = np.hstack([obs_next, [r]]) # add previous reward
                    if include_prev_action:
                        a_prev = np.zeros(env.action_space.n); a_prev[a] = 1.
                        obs_next = np.hstack([obs_next, a_prev]) # add previous action

                    # save
                    trial.update(obs, a, r, h.numpy(), q.numpy(), info.get('state', None))
                    obs = obs_next
                trials.append(trial)
    return trials
