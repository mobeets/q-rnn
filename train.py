import numpy as np
import torch
import torch.nn.functional as F
from tasks.trial import Trial

device = torch.device('cpu')
tol = np.finfo('float').min

def train(q_net=None, target_q_net=None, episode_memory=None,
          device=None, optimizer=None, batch_size=1, gamma=0.99, l2_penalty=0):

    assert device is not None, "None Device input: device should be selected."

    # Get batch from replay buffer
    samples, seq_len = episode_memory.sample()

    observations = []
    actions = []
    rewards = []
    next_observations = []
    not_dones = []

    for i in range(batch_size):
        observations.append(samples[i]["obs"])
        actions.append(samples[i]["acts"])
        rewards.append(samples[i]["rews"])
        next_observations.append(samples[i]["next_obs"])
        not_dones.append(samples[i]["not_done"])

    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_observations = np.array(next_observations)
    not_dones = np.array(not_dones)

    observations = torch.FloatTensor(observations.reshape(batch_size,seq_len,-1)).to(device)
    actions = torch.LongTensor(actions.reshape(batch_size,seq_len,-1)).to(device)
    rewards = torch.FloatTensor(rewards.reshape(batch_size,seq_len,-1)).to(device)
    next_observations = torch.FloatTensor(next_observations.reshape(batch_size,seq_len,-1)).to(device)
    not_dones = torch.FloatTensor(not_dones.reshape(batch_size,seq_len,-1)).to(device)

    with torch.no_grad():
        h_target = target_q_net.init_hidden_state(batch_size=batch_size, training=True)
        q_target, _ = target_q_net(next_observations, h_target.to(device))
        q_target_max = q_target.max(2)[0].view(batch_size,seq_len,-1).detach()
        targets = rewards + gamma*q_target_max*not_dones

    h = q_net.init_hidden_state(batch_size=batch_size, training=True)
    q_out, h_out = q_net(observations, h.to(device))
    q_a = q_out.gather(2, actions)

    # Multiply Importance Sampling weights to loss
    loss = F.mse_loss(q_a, targets)
    
    if l2_penalty > 0:
        # penalize L2 norm of RNN's activations
        loss += l2_penalty * h_out.pow(2).sum(2).mean()
    
    # Update Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss/batch_size

def prev_reward_wrapper(obs, r):
    return np.hstack([obs, [r]])

def prev_action_wrapper(obs, a, k):
    ac = np.zeros(k)
    if a is not None:
        ac[a] = 1.
    return np.hstack([obs, ac])

def beron_wrapper(obs, k):
    assert len(obs) == 4 # isi, rew, aL, aR
    new_obs = np.zeros(k)
    new_obs[-1] = obs[0] # copy initial input to end
    if obs[1] == 1 and obs[2] == 1:
        new_obs[0] = 1. # A
    elif obs[1] == 0 and obs[3] == 1:
        new_obs[1] = 1. # b
    elif obs[1] == 0 and obs[2] == 1:
        new_obs[2] = 1. # a
    elif obs[1] == 1 and obs[3] == 1:
        new_obs[3] = 1. # B
    return new_obs # A, b, a, B, isi

def beron_censor(obs, include_beron_wrapper):
    if include_beron_wrapper:
        obs[:-1] = 0. # censor all but the isi indicator
    else:
        obs[1:] = 0. # censor all but the isi indicator
    return obs

def probe_model(model, env, nepisodes, ntrials_per_episode, behavior_policy=None,
                epsilon=0, tau=tol,
                include_prev_reward=True, include_prev_action=True,
                include_beron_wrapper=False, include_beron_censor=False):
    
    trials = []
    with torch.no_grad():
        for i in range(nepisodes):
            h = model.init_hidden_state(training=False)
            if behavior_policy is not None:
                hp = behavior_policy.init_hidden_state(training=False)
            r = 0
            a = None

            for j in range(ntrials_per_episode):
                obs = np.array([env.reset()[0]])
                if include_prev_reward:
                    obs = prev_reward_wrapper(obs, r)
                if include_prev_action:
                    obs = prev_action_wrapper(obs, a, env.action_space.n)
                if include_beron_wrapper:
                    obs = beron_wrapper(obs, model.input_size)
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

                    # prepare next observation
                    if include_prev_reward:
                        obs_next = prev_reward_wrapper(obs_next, r)
                    if include_prev_action:
                        obs_next = prev_action_wrapper(obs_next, a, env.action_space.n)
                    if include_beron_wrapper:
                        obs_next = beron_wrapper(obs_next, model.input_size)
                    if include_beron_censor:
                        obs_next = beron_censor(obs_next, include_beron_wrapper)

                    # save
                    trial.update(obs, a, r, h.numpy(), q.numpy(), info.get('state', None))
                    obs = obs_next
                trials.append(trial)
    return trials
