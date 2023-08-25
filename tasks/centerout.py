import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.random import default_rng

class CenterOutEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }
    def __init__(self, ntargets=8, tau=20, distance_thresh=20, maxtimesteps=100, abort_penalty=-100, progress_weight=0, render_mode=None):
        self.ntargets = ntargets
        self.tau = tau
        self.maxtimesteps = maxtimesteps
        self.distance_thresh = distance_thresh
        self.abort_penalty = abort_penalty
        self.progress_weight = progress_weight
        self.screen_width = 800
        self.screen_height = 600
        self.target_angles = np.arange(0, 360, 360/self.ntargets)
        self.target_dist = 250

        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,)) # (target pos, hand pos)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,)) # (vx, vy)

        self.state = {}
        self.rng_state = default_rng()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def _normalize_pos(self, pos):
        low = np.zeros(2)
        high = np.array([self.screen_width, self.screen_height])
        return 2*(pos-low)/(high-low) - 1 # normalize between -1 and 1
    
    def _target_coords(self, theta):
        trgpos = np.array([self.screen_width/2, self.screen_height/2])
        trgpos += self.target_dist * np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))])
        return trgpos

    def _init_state(self, theta=None):
        theta = self.rng_state.choice(self.target_angles) if theta is None else theta
        trgpos = self._target_coords(theta)
        self.state = {
            'theta': theta,
            'target': trgpos,
            'progress': 0,
            'hand': np.array([self.screen_width/2, self.screen_height/2]),
            't': 0,
            }

    def _update_state(self, action):
        # update ball position
        self.last_state = dict(**self.state)
        trgdir = self.state['target'] - self.state['hand']
        self.state['progress'] = np.dot(trgdir / np.linalg.norm(trgdir), action / np.linalg.norm(action))
        self.state['hand'] += self.tau * (action / np.linalg.norm(action))
        self.state['t'] += 1

    def _get_obs(self):
        """ return observations, normalized between -1 and 1 """
        return np.hstack([self._normalize_pos(self.state['target']), self._normalize_pos(self.state['hand'])])

    def _get_info(self):
        return {key: val.copy() if type(val) is not int else val for key, val in self.state.items()}
    
    def _is_hit(self):
        return np.linalg.norm(self.state['target'] - self.state['hand']) < self.distance_thresh

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng_state = default_rng(seed+1)

        self._init_state(theta=options.get('theta', None) if type(options) is dict else None)
        obs = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action):
        self._update_state(action)
        obs = self._get_obs()
        info = self._get_info()
        
        terminated = False
        truncated = False
        reward = -1
        if self._is_hit():
            # hand reached target
            reward = 1.0
            terminated = True
        elif self.state['hand'][0] < 0 or self.state['hand'][0] > self.screen_width or self.state['hand'][1] < 0 or self.state['hand'][1] > self.screen_height:
            # terminate if hand moves off screen
            terminated = True
            reward = self.abort_penalty
            obs *= 0
        elif self.state['t'] > self.maxtimesteps:
            # truncate if we've reached max timesteps
            truncated = True
            reward = self.abort_penalty
        if self.progress_weight > 0:
            reward += self.progress_weight * self.state['progress']
        
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        import pygame
        from pygame import gfxdraw
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.state is None:
            return None
        
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        # draw target
        gfxdraw.filled_circle(
            self.surf,
            int(self.state['target'][0]),
            int(self.state['target'][1]),
            int(self.distance_thresh/2),
            (200, 132, 203),
        )

        # draw hand
        gfxdraw.filled_circle(
            self.surf,
            int(self.state['hand'][0]),
            int(self.state['hand'][1]),
            int(self.distance_thresh/2),
            (129, 132, 203),
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
