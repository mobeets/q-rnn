import numpy as np

def get_itis(self, ntrials=None):
    ntrials = ntrials if ntrials is not None else self.ntrials
    # note: we subtract 1 b/c 1 is the min value returned by geometric
    
    if self.iti_dist is None or self.iti_dist == 'geometric':
        itis = self.rng_iti.geometric(p=self.iti_p if self.iti_p is not None else 0.5, size=ntrials) - 1
    elif self.iti_dist == 'uniform':
        itis = self.rng_iti.choice(range(self.iti_max-self.iti_min+1), size=ntrials)
    else:
        raise Exception("Unrecognized ITI distribution")
    return self.iti_min + itis

class Trial:
    def __init__(self, state, iti=None, index_in_episode=None, episode_index=None):
        self.state = state
        self.iti = iti
        self.S = [] # states (if applicable)
        self.X = [] # inputs
        self.A = [] # actions
        self.Q = [] # state-action values
        self.Z = [] # rnn activations
        self.R = [] # rewards
        self.trial_length = 0
        self.index_in_episode = index_in_episode
        self.episode_index = episode_index
        self.tag = None

    def update(self, o, a, r, h, q, s=None):
        if len(self.X) == 0:
            self.X = o[None,:]
            self.A = np.array([a])
            self.R = np.array([r])
            self.Z = h.flatten()[None,:]
            self.Q = q.flatten()[None,:]
            self.S = np.array([s])
        else:
            self.X = np.vstack([self.X, o])
            self.A = np.hstack([self.A, [a]])
            self.Z = np.vstack([self.Z, h.flatten()])
            self.Q = np.vstack([self.Q, q.flatten()])
            self.R = np.hstack([self.R, [r]])
            self.S = np.hstack([self.S, [s]])
        self.trial_length = len(self.X)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __len__(self):
        return self.trial_length

    def __str__(self):
        return f'{self.state=}, {self.iti=}, {self.tag=}, {self.index_in_episode=}, {self.trial_length=}'
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.__str__()})'
