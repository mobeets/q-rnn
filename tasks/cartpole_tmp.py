import gymnasium as gym
from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole

class DelayedStatelessCartpole(gym.Env):
    def __init__(self, delay=0, max_timesteps_per_episode=500, **kwargs):
        self.env = StatelessCartPole()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.max_timesteps_per_episode = max_timesteps_per_episode

        self.delay = delay
        self.initial_obs = np.zeros(self.env.observation_space.shape[0])
        self.last_obs = [self.initial_obs]*self.delay

    def reset(self, seed=None, options=None):
        self.t = 0
        return self.env.reset(seed=seed, options=options)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.t += 1
        if self.t > self.max_timesteps_per_episode:
            terminated = True

        # update observation queue
        if self.delay > 0:
            cur_obs = self.last_obs.pop(0) # current obs, given delay
            self.last_obs.append(obs)
        else:
            cur_obs = obs

        return cur_obs, reward, terminated, truncated, info
