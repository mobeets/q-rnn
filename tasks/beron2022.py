import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.random import default_rng

class Beron2022_TrialLevel(gym.Env):
    def __init__(self, p_rew_max=0.7, p_switch=0.98, ntrials=1):
        self.observation_space = spaces.Discrete(1) # 0 every time
        self.action_space = spaces.Discrete(2) # left or right port
        self.p_rew_max = p_rew_max # max prob of reward; should be in [0.5, 1.0]
        self.p_switch = p_switch # prob. of state change each trial
        self.rng_state = default_rng()
        self.rng_reward = default_rng()

        self.state = None # if left (0) or right (1) port has higher reward prob
        self.trial_count = 0
        self.ntrials = ntrials

    def _get_obs(self):
        """
        returns 0 every time
        """
        return 0
    
    def _update_state(self):
        """
        flips state with probability 1-self.p_switch
        """
        if self.state is None:
            self.state = int(self.rng_state.random() < 0.5)
        do_switch_state = self.rng_state.random() > self.p_switch
        if do_switch_state:
            self.state = int(self.state == 0)

    def _sample_reward(self, state, action):
        """
        reward probability is determined by whether agent chose the high port
        """
        if state == action:
            p_reward = self.p_rew_max
        else:
            p_reward = 1-self.p_rew_max
        return int(self.rng_reward.random() < p_reward)

    def reset(self, seed=None, options=None):
        """
        start new trial
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rng_state = default_rng(seed)
            self.rng_reward = default_rng(seed+1)
        self.trial_count = 0
        self._update_state()
        observation = self._get_obs()
        return observation, None
    
    def step(self, action):
        """
        agent chooses a port
        """
        reward = self._sample_reward(self.state, action)

        self.trial_count += 1
        done = self.trial_count == self.ntrials
        info = {'state': self.state}
        self._update_state()
        if not done:
            observation = self._get_obs()
        else:
            observation = 0
        return observation, reward, done, False, info
