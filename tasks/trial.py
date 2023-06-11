import numpy as np

def get_itis(self, ntrials=None):
    ntrials = ntrials if ntrials is not None else self.ntrials
    # note: we subtract 1 b/c 1 is the min value returned by geometric
    
    if self.iti_dist == 'geometric':
        itis = np.random.geometric(p=self.iti_p, size=ntrials) - 1
    elif self.iti_dist == 'uniform':
        itis = np.random.choice(range(self.iti_max-self.iti_min+1), size=ntrials)
    else:
        raise Exception("Unrecognized ITI distribution")
    return self.iti_min + itis
