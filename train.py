import numpy as np
import torch
import torch.nn.functional as F
from tasks.trial import Trial

device = torch.device('cpu')
tol = np.finfo('float').min

def train(q_net=None, target_q_net=None, episode_memory=None,
          device=None, optimizer=None, batch_size=1,
          gamma=0.99, lmbda=0, l2_penalty=0):

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

    # MSE loss
    loss = F.mse_loss(q_a, targets)
    
    if l2_penalty > 0:
        # penalize L2 norm of RNN's activations
        loss += l2_penalty * h_out.pow(2).sum(2).mean()
    
    # Update Network
    if lmbda == 0:
        optimizer.zero_grad()
    else:
        # TD(λ)
        for p in q_net.parameters():
            if p.grad is not None:
                p.grad *= gamma*lmbda
    loss.backward()
    optimizer.step()
    
    return loss/batch_size

def probe_model(model, env, nepisodes, behavior_policy=None, epsilon=0, tau=tol, kl=None):
    
    trials = []
    with torch.no_grad():
        for i in range(nepisodes):
            h = model.init_hidden_state(training=False)
            if behavior_policy is not None:
                hp = behavior_policy.init_hidden_state(training=False)

            obs, info = env.reset()
            done = False
            if kl is not None:
                kl.reset()
            
            if hasattr(env, 'iti'):
                trial = Trial(env.state, env.iti, index_in_episode=env.trial_index, episode_index=i)
            else:
                trial = Trial(state=None, index_in_episode=env.trial_index, episode_index=i)
            while not done:   
                # get action
                cobs = torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0)
                margprefs = kl.weight*kl.marginal_pol if kl is not None else None
                if behavior_policy is None:
                    if hasattr(h, 'to'):
                        h = h.to(device)
                    a, (q, h) = model.sample_action(cobs, h, epsilon=epsilon, tau=tau, margprefs=margprefs)
                else:
                    if hasattr(h, 'to'):
                        h = h.to(device)
                        hp = hp.to(device)
                    _, (q, h) = model.sample_action(cobs, h, epsilon=epsilon, tau=tau)
                    a, (_, hp) = behavior_policy.sample_action(cobs, hp, epsilon=epsilon, tau=tau, margprefs=margprefs)

                # take action
                obs_next, r, done, truncated, info_next = env.step(a)
                r_penalty = kl.step(a, q, tau) if kl is not None else 0
                new_trial = info_next['t'] == -1

                # save
                hc = h.numpy() if hasattr(h, 'numpy') else np.array(h)
                qc = q.numpy() if hasattr(q, 'numpy') else np.array(q)
                trial.update(obs, a, r, hc, qc, info.get('state', None), r_penalty=r_penalty, margprefs=margprefs)
                if new_trial:
                    trials.append(trial)
                    if hasattr(env, 'iti'):
                        trial = Trial(env.state, env.iti, index_in_episode=env.trial_index, episode_index=i)
                    else:
                        trial = Trial(state=None, index_in_episode=env.trial_index, episode_index=i)
                obs = obs_next
                info = info_next
    return trials
