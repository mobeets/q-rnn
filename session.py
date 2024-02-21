import json
import numpy as np
import torch
from train import probe_model
from model import DRQN
from tasks.beron2022 import Beron2022, Beron2022_TrialLevel, BeronCensorWrapper, BeronWrapper
from tasks.wrappers import PreviousRewardWrapper, PreviousActionWrapper, KLMarginal
from analyze import add_beliefs_beron2022
device = torch.device('cpu')

def eval_model(model_file, ntrials, epsilon=None, tau=None, verbose=False):
    args = json.load(open(model_file))
    env_params = {
        'p_rew_max': args.get('p_reward_max', 0.8),
        'p_switch': args.get('p_switch', 0.02),
        'ntrials': ntrials}
    hidden_size = args['hidden_size']
    modelfile = args['filenames']['weightsfile_final']
    initial_modelfile = args['filenames']['weightsfile_initial']
    if verbose:
        print('H={}, prew={}, pswitch={}'.format(hidden_size, env_params['p_rew_max'], env_params['p_switch']))

    if args['experiment'] == 'beron2022_time':
        env_params.update({'iti_min': args.get('iti_min', 0), 'iti_p': args.get('iti_p', 0.5),
                        'iti_dist': args.get('iti_dist', 'geometric'), 'iti_max': args.get('iti_max', 0), 
                        'abort_penalty': args.get('abort_penalty', 0),
                        'reward_delay': args.get('reward_delay', 0),
                        'jitter': 0,#args.get('jitter', 0),
                        'include_null_action': args.get('abort_penalty', 0) < 0})
        print(env_params)
        env = Beron2022(**env_params)
    else:
        env = Beron2022_TrialLevel(**env_params)

    input_size = 1 + args['include_prev_reward'] + args['include_prev_action']*env.action_space.n
    if args['experiment'] == 'beron2022_time':
        input_size += args.get('include_beron_wrapper', False)

    if args['include_prev_reward']:
        env = PreviousRewardWrapper(env)
    if args['include_prev_action']:
        env = PreviousActionWrapper(env, env.action_space.n)
    if args['include_beron_wrapper']:
        env = BeronWrapper(env, input_size)
    if args.get('include_beron_censor', False):
        env = BeronCensorWrapper(env, args['include_beron_wrapper'])

    model = DRQN(input_size=input_size, # empty + prev reward + prev actions
                    hidden_size=hidden_size,
                    output_size=env.action_space.n,
                    recurrent_cell=args.get('recurrent_cell', 'gru')).to(device)
    model.load_weights_from_path(modelfile)

    if args.get('kl_penalty', 0) > 0:
        kl = KLMarginal(args['kl_penalty'], args['margpol_alpha'], env.action_space.n, args['include_prev_reward'])
    else:
        kl = None

    # behavior_policy = DRQN(input_size=input_size, # empty + prev reward + prev actions
    #                 hidden_size=hidden_size,
    #                 output_size=env.action_space.n,
    #                 recurrent_cell=args.get('recurrent_cell', 'gru')).to(device)
    # behavior_policy.load_weights_from_path(initial_modelfile)
    behavior_policy = None

    # probe model
    Trials = {}
    Trials_rand = {}
    for useRandomModel in [True, False]:
        if useRandomModel:
            # model.reset(gain=1)
            model.load_weights_from_path(initial_modelfile)
        else:
            model.load_weights_from_path(modelfile)
        
        for name, seed in {'train': 456, 'test': 787}.items():
            # reset seeds
            env.state = None
            env.reset(seed=seed)
            model.reset_rng(seed+1)
            if behavior_policy is not None:
                behavior_policy.reset_rng(seed+2)

            # run model on trials
            trials = probe_model(model, env, behavior_policy=behavior_policy,
                                    epsilon=epsilon, tau=tau, nepisodes=1, kl=kl)
            if verbose:
                print(useRandomModel, name, np.round(np.hstack([trial.R for trial in trials]).mean(),3))

            # add beliefs
            add_beliefs_beron2022(trials, env.p_rew_max, env.p_switch)
            ymx = np.max(np.abs(np.hstack([trial.Q[:,1]-trial.Q[:,0] for trial in trials])))
            for trial in trials:
                trial.Qdiff = (trial.Q[:,1] - trial.Q[:,0])[:,None]/ymx
            
            # discard trials at the beginning of episodes (warmup)
            trials = [trial for trial in trials if trial.index_in_episode > 5]

            if useRandomModel:
                Trials_rand[name] = trials
            else:
                Trials[name] = trials
    return Trials, Trials_rand, model, env
