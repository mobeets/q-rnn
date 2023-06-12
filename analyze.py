import numpy as np

def add_beliefs(trials, p_iti, p_coh):
    # assert env.iti_min == 0
    O = np.zeros((3,3))
    O[:,0] = [p_coh, 0, 1-p_coh] # P(O = o | S = -1)
    O[:,1] = [0, 1, 0] # P(O = o | S = 0)
    O[:,2] = [1-p_coh, 0, p_coh] # P(O = o | S =  1)

    T = np.zeros((3,3,3))
    T[:,:,0] = np.array([0,1,0])
    T[:,:,0] = np.tile(np.array([0,1,0])[:,None], (1,3)) # P(S' = s' | S = s, A = 0)
    T[:,:,1] = np.tile(np.array([0,1,0])[:,None], (1,3)) # P(S' = s' | S = s, A = 1)
    T[:,:,2] = np.eye(3) # P(S' = s' | S = s, A = 2)
    T[:,1,2] = [p_iti/2, 1-p_iti, p_iti/2] # transitions out of ITI

    b_init = np.array([0,1,0]) # belief at start of episode
    b_prev = b_init

    B = []
    for trial in trials:
        B_trial = []
        if trial.index_in_episode is None or trial.index_in_episode == 0:
            b_prev = b_init
            a_prev = 2
        for x,a in zip(trial.X, trial.A):
            p_obs = O[x[0]+1,:]
            p_tra = (T[:,:,a_prev] @ b_prev)
            b = p_obs * p_tra
            b = b/b.sum()

            a_prev = a
            b_prev = b
            B.append(b)
            B_trial.append(b)
        trial.B = np.vstack(B_trial)
    B = np.vstack(B)
    return B, (O, T)
