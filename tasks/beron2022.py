import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.random import default_rng
from tasks.trial import get_itis

class Beron2022(gym.Env):
    def __init__(self, p_rew_max=0.8, p_switch=0.02,
                 iti_min=0, iti_p=0.25, iti_max=0, iti_dist='geometric',
                 include_null_action=False, abort_penalty=0):
        self.observation_space = spaces.Discrete(1) # 0 every time
        self.action_space = spaces.Discrete(2 + include_null_action) # left port, right port, null [optional]
        self.include_null_action = include_null_action
        self.p_rew_max = p_rew_max # max prob of reward; should be in [0.5, 1.0]
        self.p_switch = p_switch # prob. of state change each trial
        self.state = None # if left (0) or right (1) port has higher reward prob
        self.abort_penalty = abort_penalty # penalty for acting during ITI (if include_null_action==True)
        if not self.include_null_action and self.abort_penalty != 0:
            raise Exception("Cannot provide a nonzero abort penalty if there is no null action")

        self.iti_min = iti_min
        self.iti_max = iti_max # n.b. only used if iti_dist == 'uniform'
        self.iti_p = iti_p
        self.iti_dist = iti_dist
        
        self.rng_state = default_rng()
        self.rng_reward = default_rng()
        self.rng_iti = default_rng()

    def _get_obs(self):
        """
        returns 0 every time
        """
        if self.t < self.iti:
            return 0
        return 1
    
    def _update_state(self):
        """
        flips state with probability self.p_switch
        """
        if self.state is None:
            self.state = int(self.rng_state.random() < 0.5)
        do_switch_state = self.rng_state.random() < self.p_switch
        if do_switch_state:
            self.state = int(self.state == 0) # 0 -> 1, and 1 -> 0

    def _sample_reward(self, state, action):
        """
        reward probability is determined by whether agent chose the high port
        """
        if self.t < self.iti:
            if action == 2:
                return self.abort_penalty
            else:
                return 0
        elif action == 2: # no decision yet
            return 0 
        else: # decision report
            if state == action:
                p_reward = self.p_rew_max
            elif action < 2:
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
            self.rng_iti = default_rng(seed+2)
        
        self.t = -1 # -1 to ensure we get at least one ITI observation between trials
        self.iti = get_itis(self, ntrials=1)[0]
        self._update_state()
        observation = self._get_obs()
        return observation, None
    
    def step(self, action):
        """
        agent chooses a port
        """
        done = self.t >= self.iti
        reward = self._sample_reward(self.state, action)
        info = {'state': self.state, 'iti': self.iti, 't': self.t}
        self.t += 1
        if not done:
            observation = self._get_obs()
        else:
            observation = 0
        return observation, reward, done, False, info

class Beron2022_TrialLevel(gym.Env):
    def __init__(self, p_rew_max=0.8, p_switch=0.02):
        self.observation_space = spaces.Discrete(1) # 0 every time
        self.action_space = spaces.Discrete(2) # left or right port
        self.p_rew_max = p_rew_max # max prob of reward; should be in [0.5, 1.0]
        self.p_switch = p_switch # prob. of state change each trial
        self.rng_state = default_rng()
        self.rng_reward = default_rng()
        self.state = None # if left (0) or right (1) port has higher reward prob

    def _get_obs(self):
        """
        returns 0 every time
        """
        return 0
    
    def _update_state(self):
        """
        flips state with probability self.p_switch
        """
        if self.state is None:
            self.state = int(self.rng_state.random() < 0.5)
        do_switch_state = self.rng_state.random() < self.p_switch
        if do_switch_state:
            self.state = int(self.state == 0) # 0 -> 1, and 1 -> 0

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
        self._update_state()
        observation = self._get_obs()
        return observation, None
    
    def step(self, action):
        """
        agent chooses a port
        """
        reward = self._sample_reward(self.state, action)
        done = True
        info = {'state': self.state}
        observation = 0
        return observation, reward, done, False, info
