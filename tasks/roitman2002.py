import gymnasium as gym
from gymnasium import spaces
from tasks.trial import get_itis
from numpy.random import default_rng

class Roitman2002(gym.Env):
    def __init__(self, reward_amounts, p_coh=0.6, fixed_response_time=None,
                 iti_min=0, iti_p=0.5, iti_max=0, iti_dist='geometric', ntrials=100):
    
        self.reward_amounts = reward_amounts # Correct, Incorrect, Abort, Wait
        assert len(self.reward_amounts) == 4
        self.observation_space = spaces.Discrete(3, start=-1) # Left, Null, Right
        self.action_space = spaces.Discrete(3) # Left, Right, or Wait
        self.p_coh = p_coh # should be in [0.5, 1.0]
        if self.p_coh < 0.5 or self.p_coh > 1.0:
            raise Exception("p_coh must be in [0.5, 1.0]")
        self.fixed_response_time = fixed_response_time
        self.iti_min = iti_min
        self.iti_max = iti_max # n.b. only used if iti_dist == 'uniform'
        self.iti_p = iti_p
        self.iti_dist = iti_dist

        self.ntrials = ntrials
        self.trial_index = None
        
        self.rng_state = default_rng()
        self.rng_obs = default_rng()
        self.rng_iti = default_rng()

    def _get_obs(self):
        """
        returns observation coherent or incoherent with current state
            e.g., if state == 1, coherent observation is obs=1, incoherent is obs=-1
        """
        if self.t < self.iti:
            return 0
        is_coherent = (self.rng_obs.random() <= self.p_coh)
        return self.state if is_coherent else -self.state
    
    def _get_info(self):
        return {'state': self.state, 'iti': self.iti, 't': self.t, 'trial_index': self.trial_index}
    
    def _update_state(self):
        self.state = 2*int(self.rng_state.random() > 0.5) - 1 # -1 or 1

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
            self.rng_obs = default_rng(seed+1)
            self.rng_iti = default_rng(seed+2)
        self.state = None
        self.trial_index = -1
        
        self._new_trial()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        if self.fixed_response_time is None:
            trial_done = action != 2 # trial ends when decision is made
    
            if action == 2: # wait
                reward = self.reward_amounts[-1]
            elif self.t < self.iti: # action prior to stim onset aborts trial
                reward = self.reward_amounts[2]
            elif (2*action-1) == self.state: # correct decision
                reward = self.reward_amounts[0]
            else: # incorrect decision
                reward = self.reward_amounts[1]
        else:
            trial_done = self.t - self.iti >= self.fixed_response_time

            if trial_done:
                if (2*action-1) == self.state: # correct decision
                    reward = self.reward_amounts[0]
                else: # incorrect decision
                    reward = self.reward_amounts[1]
        
        done = trial_done and (self.trial_index+1 >= self.ntrials)
        if not done:
            if trial_done:
                self._new_trial()
            else:
                self.t += 1
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, done, False, info
