import numpy as np
import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.spaces import Box, Discrete

class PreviousActionWrapper(Wrapper):
    """
    appends the previous action (as a one-hot vector) to the current observation
    """
    def __init__(self, env, nactions):
        super().__init__(env)
        self.observation_space = None # todo
        self.nactions = nactions
        self.last_action = None
    
    def observation(self, obs):
        ac = np.zeros(self.nactions)
        if self.last_action is not None:
            ac[self.last_action] = 1.
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

class PreviousRewardWrapper(Wrapper):
    """
    appends the previous reward to the current observation
    """
    def __init__(self, env, initial_prev_reward=0):
        super().__init__(env)
        self.observation_space = None # todo
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
