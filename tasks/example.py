import gymnasium as gym
from gymnasium import spaces
from numpy.random import default_rng
from tasks.trial import get_itis

class ExampleNABT(gym.Env):
    """
    A simple N-armed bandit task (NABT) modeled at the time step level
        where decisions are separated by a random intertrial interval
    """
    def __init__(self, ntrials, reward_probs=(0.8, 0.4)):
        self.observation_space = spaces.Discrete(2) # 0 during ITI, 1 during ISI
        self.action_space = spaces.Discrete(len(reward_probs)) # action choices
        self.reward_probs = reward_probs # reward probabilities for each action
        self.ntrials = ntrials # total number of trials in episode
        self.trial_index = None # for keeping track of trial index within episode
        self.t = None # for keeping track of timestep index within trial
        self.rng = default_rng() # random number generator

    def _new_trial(self):
        """
        start new trial
        """
        if self.trial_index is None:
            self.trial_index = 0
        else:
            self.trial_index += 1 # increment trial counter
        self.t = -1 # initialize to -1 to ensure we get at least one ITI observation between trials
        self.iti = get_itis(self, ntrials=1)[0] # sample the ITI for the current trial

    def reset(self, seed=None, options=None):
        """
        start new episode
        """
        # set any random seeds
        super().reset(seed=seed)
        if seed is not None:
            self.rng = default_rng(seed)
        
        self._new_trial() # create trial
        observation = self._get_obs() # get current observation
        info = self._get_info() # get current trial info
        return observation, info
    
    def _get_obs(self):
        """
        returns indicator of whether or not we are in the ISI
        """
        return 0 if self.t < self.iti else 1
    
    def _get_info(self):
        return {'t': self.t, 'trial_index': self.trial_index}
    
    def _sample_reward(self, action):
        """
        returns reward with probability determined by the chosen arm
        """
        return int(self.rng.random() < self.reward_probs[action])
    
    def step(self, action):
        """
        agent chooses a port and receives reward
        """
        trial_done = (self.t >= self.iti) # trial is done when we have had one ISI time step
        done = (self.trial_index+1 >= self.ntrials) and trial_done # episode is done when the nth trial is done
        reward = self._sample_reward(action)
        if not done:
            if trial_done:
                self._new_trial()
            else:
                self.t += 1
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, done, False, info
