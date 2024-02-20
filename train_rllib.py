import os
import time
from datetime import datetime
import numpy as np
import ray
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger
from model_rllib import build_agent
from tasks.beron2022 import Beron2022, Beron2022_TrialLevel

os.environ['CUDA_VISIBLE_DEVICES'] ='2'
NUM_CPUS = os.cpu_count()

def log_exp_info(config, env_config, logdir):
    with open(f"{logdir}/exp_info.log", 'w') as f:
        f.write("[--] gamma = {}\n".format(config.gamma))
        f.write("[--] learning rate = {}\n".format(config.lr))
        f.write("[--] dueling = {}\n".format(config.dueling))
        f.write("[--] double q = {}\n".format(config.double_q))
        f.write("[++] Config model:\n")
        for (k, v) in config.model.items():
            f.write("\t[--] {} = {}\n".format(k, v))
        f.write("[++] Exploration strategy:\n")
        for (k, v) in config.exploration_config.items():
            f.write("\t[--] {} = {}\n".format(k, v))
        f.write("[++] Environment configuration:\n")
        for (k, v) in env_config.items():
            f.write("\t[--] {} = {}\n".format(k, v))
        f.flush()

def save_checkpoint(algo, logdir, i=0):
    if logdir is None:
        return
    outdir = f"{logdir}/checkpoints/checkpoint_{i}"
    os.makedirs(outdir, exist_ok=True)
    algo.save(outdir)
    return outdir

def make_algo(env_name, config, env_config, logdir=None, gamma=0.9, lr=0.1, eps_init=0.7, eps_final=0.1, eps_timesteps=5000, num_gpus=0):
    config.environment(env=env_name, env_config=env_config)
    config.lr = lr
    config.gamma = gamma
    config.exploration_config['initial_epsilon'] = eps_init
    config.exploration_config['final_epsilon'] = eps_final
    config.exploration_config['epsilon_timesteps'] = eps_timesteps
    config.evaluation_duration = 1; config.evaluation_interval = 1 # evaluate using 1 episode every episode
    if num_gpus > 0:
        config = config.resources(num_gpus=num_gpus)
        # force rollouts to be done by trainer actor (not sure why marco did this):
        config = config.rollouts(num_rollout_workers=0)

    logger_creator = None
    if logdir is not None:
        logger_creator = lambda config: UnifiedLogger(config, logdir=logdir)
    algo = config.build(logger_creator=logger_creator)
    return algo, config

def train(algo, nepisodes=200, logdir=None):
    outputs = []
    best_score = -np.inf
    best_checkpoint = None
    get_score = lambda x: x['evaluation']['episode_reward_mean'] if 'evaluation' in x else x['episode_reward_mean']

    save_checkpoint(algo, logdir, 'init')
    print("Training....")
    for i in range(nepisodes):
        tr_time = time.time()
        output = algo.train()
        tr_time = time.time() - tr_time
        outputs.append(output)        
        score = get_score(outputs[-1])
        epsilons = algo.workers.foreach_worker(lambda worker: worker.get_policy().exploration.get_state()['cur_epsilon'])
        
        print('{}. Score={:0.3f}, Best={:0.3f}, Time={:0.3f}, Epsilons={}'.format(i, score, best_score, tr_time, epsilons))

        if score > best_score or i == nepisodes-1:
            # note: this isn't always the best model...
            best_score = score
            best_checkpoint = save_checkpoint(algo, logdir, i)
    print("Training complete.")
    print('Path to best model: '.format(best_checkpoint))
    return outputs

def main(args):
    # create environment
    if args.experiment == 'beron2022_time':
        Env = Beron2022
        env_config = {'p_rew_max': 0.8, 'p_switch': 0.02, 'ntrials': 800}
    else:
        raise Exception(f"Unrecognized env_name: {args.experiment}")
    register_env(args.experiment, lambda env_config: Env(**env_config))

    # initialize ray, logging, and agent
    ray.init(num_cpus=NUM_CPUS)
    config = build_agent(args.latent_size, max_seq_len=args.max_seq_len, use_prev_action=args.use_prev_action, use_prev_reward=args.use_prev_reward)
    logdir = os.path.join(args.savedir, args.run_name + datetime.now().strftime('_%Y%m%d-%H%M%S'))
    os.makedirs(logdir, exist_ok=True)
    algo = make_algo(args.experiment, config, env_config, logdir=logdir, gamma=args.gamma, lr=args.lr, eps_init=args.eps_init, eps_final=args.eps_final, eps_timesteps=args.eps_timesteps, num_gpus=0)
    log_exp_info(config, env_config, logdir)

    # train model
    outputs = train(algo, args.episodes, logdir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, help='tag for current run', default='')
    parser.add_argument('-d', '--save_dir', type=str,
                        default='data/models', help='where to save trained model')
    parser.add_argument('--experiment', type=str,
                        default='beron2022_time', choices=['beron2022_time'])
    parser.add_argument('-k', '--hidden_size', type=int,
                        default=10, help='number of hidden units in the rnn')
    parser.add_argument('--max_seq_len', type=int,
                        default=300, help='number of time steps used in BPTT')
    parser.add_argument('-g', '--gamma', type=float,
                        default=0.9, help='reward discount factor')
    parser.add_argument('--lr', type=float,
                        default=0.001, help='learning rate')
    parser.add_argument('--eps_init', type=float,
                        default=0.5, help='initial epsilon for exploration')
    parser.add_argument('--eps_final', type=float,
                        default=0.001, help='final epsilon for exploration')
    parser.add_argument('--eps_timesteps', type=int,
                        default=5000, help='number of timesteps to decay epsilon')    
    parser.add_argument('-e', '--episodes', type=int,
                        default=200, help='total number of episodes to train on')
    parser.add_argument('--include_prev_reward', action='store_true',
                        default=False)
    parser.add_argument('--include_prev_action', action='store_true',
                        default=False)
    args = parser.parse_args()
    if 'beron2022' in args.experiment:
        print('WARNING: For {}, auto-including prev reward and action.\n'.format(args.experiment))
        args.include_prev_reward = True
        args.include_prev_action = True
    main(args)
