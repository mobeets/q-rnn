import numpy as np

def lsql(experience, gamma, k=None, d=None):
    """
    source: "Least-Squares Methods in Reinforcement Learning for Control" Lagoudakis et al. (2002)
    """
    if k is None:
        k = len(np.unique([s for s,a,r,sp in experience])) # number of states
    if d is None:
        d = len(np.unique([a for s,a,r,sp in experience])) # number of actions
    A = np.zeros((k*d,k*d))
    b = np.zeros((k*d,1))
    w = np.zeros((k*d,1))

    def zf(s,a):
        z = np.zeros((k,d))
        z[s,a] = 1.
        return z.flatten()[:,None]

    for t, (s,a,r,sprime) in enumerate(experience):
        z = zf(s,a)
        Ac = z @ z.T
        if gamma > 0:
            aprime = np.argmax(w.reshape(k,d)[sprime,:])
            zp = zf(sprime,aprime)
            bc = z*(r + gamma*(zp.T @ w))
        else:
            bc = z*r

        alpha = 1/(t+1)
        A = (1-alpha)*A + alpha*Ac
        b = (1-alpha)*b + alpha*bc
        if gamma > 0:
            w = np.linalg.lstsq(A, b, rcond=None)[0]

    w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w.reshape(k,d)

def belief_fixed_points_beron2022(p_rew_max, p_switch, niters=100):
    Bs = {}
    b_inits = [0, 0.25, 0.5, 0.75, 1]
    for a_prev in [0,1]:
        for r_prev in [0,1]:
            for b_init in b_inits:
                b_prev = b_init
                B = []
                for _ in range(niters):
                    # b(t) = P(s(t) = 1 | a(1:t-1), r(1:t-1))
                    #      = P(s(t) = 1 | a(t-1), r(t-1), b(t-1))
                    b_lik_0 = (1-p_rew_max) if a_prev == r_prev else p_rew_max # P(r | s=0, a)
                    b_lik_1 = p_rew_max if a_prev == r_prev else (1-p_rew_max) # P(r | s=1, a)
                    b_lik = b_prev*b_lik_1 / ((1-b_prev)*b_lik_0 + b_prev*b_lik_1)
                    b = p_switch*(1-b_lik) + (1-p_switch)*b_lik

                    B.append(b)
                    b_prev = b
                Bs[(a_prev, r_prev, b_init)] = B
    return Bs, b_inits

def add_beliefs_beron2022(trials, p_rew_max, p_switch, b_init=0.5):
    B = []
    b_prev = b_init
    for trial in trials:
        if not hasattr(trial, 'index_in_episode') or trial.index_in_episode is None or trial.index_in_episode == 0:
            # b(0) = b_prior
            b = b_init
        else:
            # b(t) = P(s(t) = 1 | a(1:t-1), r(1:t-1))
            #      = P(s(t) = 1 | a(t-1), r(t-1), b(t-1))
            b_lik_0 = (1-p_rew_max) if a_prev == r_prev else p_rew_max # P(r | s=0, a)
            b_lik_1 = p_rew_max if a_prev == r_prev else (1-p_rew_max) # P(r | s=1, a)
            b_lik = b_prev*b_lik_1 / ((1-b_prev)*b_lik_0 + b_prev*b_lik_1)
            b = p_switch*(1-b_lik) + (1-p_switch)*b_lik

        trial.B = np.array([b]*len(trial.A))[:,None]
        B.append(b)
        b_prev = b
        a_prev = trial.A[-1]
        r_prev = trial.R[-1]

    B = np.hstack(B)[:,None]
    O = None
    T = None
    return B, (O, T)

def add_beliefs_roitman2002(trials, p_iti, p_coh):
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
