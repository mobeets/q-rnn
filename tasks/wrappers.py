import numpy as np
from gymnasium import Wrapper, spaces
import torch.nn.functional as F

class PreviousActionWrapper(Wrapper):
    """
    appends the previous action (as a one-hot vector) to the current observation
    """
    def __init__(self, env, nactions):
        super().__init__(env)
        obs = env.observation_space
        if type(obs) is spaces.Discrete:
            self.observation_space = spaces.Box(low=np.array([0] + [0]*nactions),
                                                high=np.array([obs.n] + [1]*nactions))
        elif type(obs) is spaces.Box:
            self.observation_space = spaces.Box(low=np.array(list(obs.low) + [0]*nactions),
                                                high=np.array(list(obs.high) + [1]*nactions))
        else:
            raise Exception("Need to implement observation_space for PreviousActionWrapper")
        self.nactions = nactions
        self.last_action = None
    
    def observation(self, obs):
        ac = np.zeros(self.nactions)
        if self.last_action is not None:
            ac[self.last_action] = 1
        return np.hstack([obs, ac])
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_action = None
        return self.observation(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_action = action
        new_obs = self.observation(obs)
        return new_obs, reward, terminated, truncated, info

class KLMarginal:
    def __init__(self, weight, margpol_alpha, nactions, min_penalty=-2, max_penalty=2):
        if weight < 0:
            raise Exception("KL penalty should be positive")
        if margpol_alpha <= 0 or margpol_alpha > 1:
            raise Exception("margpol alpha should be between 0 and 1 (as in exponential smoothing)")
        self.weight = weight
        self.alpha = margpol_alpha
        self.nactions = nactions
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty

    def reset(self):
        self.marginal_pol = np.ones(self.nactions,)/self.nactions

    def step(self, action, q, tau, margpol_in_policy=True):
        # calculate penalty
        # todo: add in margpol here as well
        if margpol_in_policy:
            pol = F.softmax(q.detach()/tau + self.weight*self.marginal_pol, dim=-1).numpy().flatten()
        else:
            pol = F.softmax(q.detach()/tau, dim=-1).numpy().flatten()
        r_penalty = np.log(pol[action]) - np.log(self.marginal_pol[action])

        # constrain penalty within bounds
        if self.min_penalty is not None and r_penalty < self.min_penalty:
            r_penalty = self.min_penalty
        elif self.max_penalty is not None and r_penalty > self.max_penalty:
            r_penalty = self.max_penalty

        # update marginal policy (using exponential smoothing)
        self.marginal_pol = (1-self.alpha)*self.marginal_pol + self.alpha*pol

        return self.weight*r_penalty

class PreviousRewardWrapper(Wrapper):
    """
    appends the previous reward to the current observation
    """
    def __init__(self, env, initial_prev_reward=0):
        super().__init__(env)
        if type(env.observation_space) is spaces.Discrete:
            self.observation_space = spaces.Box(low=np.array([0,-np.inf]),
                                                high=np.array([env.observation_space.n, np.inf]),
                                                shape=(1 + 1,))
        else:
            raise Exception("Need to implement observation_space for PreviousRewardWrapper")
        self.initial_prev_reward = initial_prev_reward
        self.last_reward = initial_prev_reward
    
    def observation(self, obs):
        return np.hstack([obs, [self.last_reward]])
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_reward = self.initial_prev_reward
        return self.observation(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_reward = reward
        new_obs = self.observation(obs)
        return new_obs, reward, terminated, truncated, info

class DelayWrapper(Wrapper):
    """
    delays the observations by a given number of time steps
    """
    def __init__(self, env, delay, initial_obs=None):
        super().__init__(env)
        self.delay = delay
        if initial_obs is None:
            initial_obs = np.zeros(env.observation_space.n)
        self.initial_obs = initial_obs
        self.last_obs = [self.initial_obs]*self.delay

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # update observation queue
        if self.delay > 0:
            cur_obs = self.last_obs.pop(0) # current obs, given delay
            self.last_obs.append(obs)
        else:
            cur_obs = obs

        return cur_obs, reward, terminated, truncated, info
