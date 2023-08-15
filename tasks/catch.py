import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.random import default_rng

class CatchEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }
    def __init__(self, tau=0.01, render_mode=None):
        self.gravity = 9.8
        self.action_space = spaces.Discrete(3) # up, down, stay
        self.screen_width = 800
        self.screen_height = 600
        self.hand_length = 50
        self.hand_step = 15
        self.tau = tau # seconds between state updates
        self.observation_space = spaces.Box(low=np.array([0, 0, self.screen_width, 0]),
            high=np.array([self.screen_width, self.screen_height, self.screen_width, self.screen_height])) # ball and hand positions
        self.state = {}
        self.rng_state = default_rng()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def _init_state(self):
        obs = self.observation_space.sample()
        obs[0] = 0 # ball starts on left of screen
        obs[1] = self.screen_height/2 # ball starts halfway up
        obs[2] = self.screen_width # hand is on right of screen
        obs[3] = self.screen_height/2 # hand starts halfway up
        
        r, theta = self.rng_state.random(2)
        r_max = 1600
        r_min = 1500
        r = r*(r_max-r_min) + r_min
        # r /= 2
        theta = 40*theta # angle will be between 0 and 40
        self.state = {
            'ball': obs[:2],
            'hand': obs[2:],
            'vel': r*np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))]),
            'accel': np.array([0, -200*self.gravity]),
            't': 0,
            }

    def _update_state(self, action):
        # update ball position
        self.last_state = dict(**self.state)
        self.state['ball'] += self.tau * self.state['vel']
        self.state['vel'] += self.tau * self.state['accel']    
        self.state['t'] += 1

        # update hand position
        if action != 2:
            self.state['hand'][1] += self.hand_step*(2*action - 1)
        self.constrain_hand()

    def constrain_hand(self):
        if self.state['hand'][1] < 0:
            self.state['hand'][1] = 0
        elif self.state['hand'][1] > self.screen_height:
            self.state['hand'][1] = self.screen_height

    def _get_info(self):
        return self.state
    
    def _get_obs(self):
        return np.hstack([self.state['ball'], self.state['hand']])
    
    def _is_hit(self):
        if self.state['ball'][0] < self.state['hand'][0]-1:
            return False
        return (self.state['ball'][1] >= self.state['hand'][1]-self.hand_length/2) and (self.state['ball'][1] <= self.state['hand'][1]+self.hand_length/2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng_state = default_rng(seed+1)
        self._init_state()
        obs = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return obs, info
    
    def step(self, action):
        self._update_state(action)
        if self.state['ball'][0] >= self.state['hand'][0]:
            caught = self._is_hit()
            reward = 1.0 if caught else -1.0
            terminated = True
        else:
            reward = 0.0
            terminated = False
        truncated = False

        obs = self._get_obs()
        info = self._get_info()
        
        if self.state['ball'][0] < 0 or self.state['ball'][0] > self.screen_width or self.state['ball'][1] < 0 or self.state['ball'][1] > self.screen_height:
            terminated = True
            obs *= 0; obs[2] = self.screen_width

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

        # draw ball
        gfxdraw.filled_circle(
            self.surf,
            self.state['ball'][0],
            self.state['ball'][1],
            10,
            (129, 132, 203),
        )

        # draw hand
        hand_pos = self.state['hand'].astype(int)
        ul = (hand_pos[0]-5, hand_pos[1]+self.hand_length)
        ur = (hand_pos[0]+5, hand_pos[1]+self.hand_length)
        br = (hand_pos[0]+5, hand_pos[1]-self.hand_length)
        bl = (hand_pos[0]-5, hand_pos[1]-self.hand_length)
        lcolor = (250, 0, 0) if self._is_hit() else (0, 0, 0)
        gfxdraw.filled_polygon(self.surf, (ul, ur, br, bl), lcolor)

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
