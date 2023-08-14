import gymnasium as gym
from gymnasium import spaces
from gymnasium import ObservationWrapper
import numpy as np
from numpy.random import default_rng
from tasks.trial import get_itis

class Beron2022(gym.Env):
    def __init__(self, p_rew_max=0.8, p_switch=0.02, ntrials=100,
                 iti_min=0, iti_p=0.25, iti_max=0, iti_dist='geometric',
                 include_null_action=False, abort_penalty=0):
        self.observation_space = spaces.Discrete(2) # 0 during ITI, 1 during ISI
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

        self.ntrials = ntrials # total number of trials in episode
        self.trial_index = None
        
        self.rng_state = default_rng()
        self.rng_reward = default_rng()
        self.rng_iti = default_rng()
    
    def _update_state(self):
        """
        flips state with probability self.p_switch
        """
        if self.state is None:
            self.state = int(self.rng_state.random() < 0.5)
        do_switch_state = self.rng_state.random() < self.p_switch
        if do_switch_state:
            self.state = int(self.state == 0) # 0 -> 1, and 1 -> 0

    def _get_obs(self):
        """
        returns indicator of whether or not we are in the ISI
        """
        if self.t < self.iti:
            return 0
        return 1

    def _sample_reward(self, state, action):
        """
        reward probability is determined by whether agent chose the high port
        """
        if action == 2: # no decision yet
            return 0 
        elif self.t < self.iti: # early decision (abort penalty)
            return self.abort_penalty
        else: # decision reported on time
            if state == action:
                p_reward = self.p_rew_max
            elif action < 2:
                p_reward = 1-self.p_rew_max
            return int(self.rng_reward.random() < p_reward)
        
    def _get_info(self):
        return {'state': self.state, 'iti': self.iti, 't': self.t, 'trial_index': self.trial_index}
    
    def _new_trial(self):
        self.trial_index += 1
        self.t = -1 # -1 to ensure we get at least one ITI observation between trials
        self.iti = get_itis(self, ntrials=1)[0]
        self._update_state()

    def reset(self, seed=None, options=None):
        """
        start new episode
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rng_state = default_rng(seed)
            self.rng_reward = default_rng(seed+1)
            self.rng_iti = default_rng(seed+2)
        
        self.state = None
        self.trial_index = -1
        self._new_trial()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        """
        agent chooses a port
        """
        trial_done = (self.t >= self.iti)
        done = trial_done and (self.trial_index+1 >= self.ntrials)
        reward = self._sample_reward(self.state, action)
        if not done:
            if trial_done:
                self._new_trial()
            else:
                self.t += 1
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, done, False, info

class BeronCensorWrapper(ObservationWrapper):
    """
    censors the previous action and reward during a trial
    """
    def __init__(self, env, include_beron_wrapper):
        super().__init__(env)
        self.include_beron_wrapper = include_beron_wrapper
    
    def observation(self, obs):
        if self.env.t >= 0:
            if self.include_beron_wrapper:
                obs[:-1] = 0. # censor all but the isi indicator
            else:
                obs[1:] = 0. # censor all but the isi indicator
        return obs

class BeronWrapper(ObservationWrapper):
    """
    re-encodes the observations to indicate whether action and reward were coherent or not
    """
    def __init__(self, env, k):
        super().__init__(env)
        self.observation_space = None # todo
        self.k = k
    
    def observation(self, obs):
        assert len(obs) in [4,5] # isi, rew, aL, aR, aWait[optional]
        new_obs = np.zeros(self.k)
        new_obs[-1] = obs[0] # copy initial input to end
        if obs[1] == 1 and obs[2] == 1:
            new_obs[0] = 1. # A
        elif obs[1] == 0 and obs[3] == 1:
            new_obs[1] = 1. # b
        elif obs[1] == 0 and obs[2] == 1:
            new_obs[2] = 1. # a
        elif obs[1] == 1 and obs[3] == 1:
            new_obs[3] = 1. # B
        elif len(obs) == 5:
            assert self.k == 6
            new_obs[4] = obs[4] # wait
        return new_obs # A, b, a, B, wait[optional], isi

class Beron2022_TrialLevel(gym.Env):
    def __init__(self, p_rew_max=0.8, p_switch=0.02, ntrials=100):
        self.observation_space = spaces.Discrete(1) # 0 every time
        self.action_space = spaces.Discrete(2) # left or right port
        self.p_rew_max = p_rew_max # max prob of reward; should be in [0.5, 1.0]
        self.p_switch = p_switch # prob. of state change each trial
        self.rng_state = default_rng()
        self.rng_reward = default_rng()
        self.ntrials = ntrials
        self.trial_index = -1
        self.t = -1
        self.state = None # if left (0) or right (1) port has higher reward prob

    def _get_obs(self):
        """
        returns 0 every time
        """
        return 0
    
    def _sample_reward(self, state, action):
        """
        reward probability is determined by whether agent chose the high port
        """
        if state == action:
            p_reward = self.p_rew_max
        else:
            p_reward = 1-self.p_rew_max
        return int(self.rng_reward.random() < p_reward)
    
    def _update_state(self):
        """
        flips state with probability self.p_switch
        """
        if self.state is None:
            self.state = int(self.rng_state.random() < 0.5)
        do_switch_state = self.rng_state.random() < self.p_switch
        if do_switch_state:
            self.state = int(self.state == 0) # 0 -> 1, and 1 -> 0

    def _get_info(self):
        return {'state': self.state, 'trial_index': self.trial_index, 't': self.t}

    def _new_trial(self):
        self.trial_index += 1
        self._update_state()

    def reset(self, seed=None, options=None):
        """
        start new trial
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rng_state = default_rng(seed)
            self.rng_reward = default_rng(seed+1)
        self.state = None
        self.trial_index = -1
        self._new_trial()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        """
        agent chooses a port
        """
        reward = self._sample_reward(self.state, action)
        done = self.trial_index+1 == self.ntrials
        self._new_trial()
        info = self._get_info()
        observation = self._get_obs()
        return observation, reward, done, False, info

def belief_step_beron2022(b_prev, r_prev, a_prev, p_rew_max, p_switch):
    # b(t) = P(s(t) = 1 | a(1:t-1), r(1:t-1))
    #      = P(s(t) = 1 | a(t-1), r(t-1), b(t-1))
    b_lik_0 = (1-p_rew_max) if a_prev == r_prev else p_rew_max # P(r | s=0, a)
    b_lik_1 = p_rew_max if a_prev == r_prev else (1-p_rew_max) # P(r | s=1, a)
    b_lik = b_prev*b_lik_1 / ((1-b_prev)*b_lik_0 + b_prev*b_lik_1)
    return p_switch*(1-b_lik) + (1-p_switch)*b_lik

class BeronBeliefAgent:
    def __init__(self, env, b_init=0.5):
        self.env = env
        self.b_init = b_init
        self.b = b_init
        self.rng_agent = default_rng()
        assert self.env.action_space.n == 2

    def reset(self, seed=None):
        if seed is not None:
            self.rng_agent = default_rng(seed)
        self.b = self.b_init

    def init_hidden_state(self, **kwargs):
        self.b = self.b_init
        return self.b
    
    def sample_action(self, obs, b_prev, epsilon=None, **kwargs):
        if hasattr(obs, 'numpy'):
            obs = obs.numpy().flatten()
        try:
            b_prev = b_prev[0]
        except TypeError:
            pass
        self.b = belief_step_beron2022(b_prev, obs[1], np.argmax(obs[2:]), self.env.p_rew_max, self.env.p_switch)
        
        if epsilon is not None and epsilon > 0 and self.rng_agent.random() < epsilon:
            a = int(self.rng_agent.random() < 0.5)
        else:
            a = int(self.b > 0.5)
        return a, ([1-self.b, self.b], [self.b])
