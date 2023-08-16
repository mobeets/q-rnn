import argparse
import pygame
from tasks.catch import CatchEnv, NormalizedInputs

def load_model(checkpoint):
    from ray.rllib.algorithms import r2d2
    from ray.tune.registry import register_env

    env_config = {}
    env_name = 'catch'
    # register_env(env_name, lambda env_config: CatchEnv(**env_config))
    register_env(env_name, lambda env_config: NormalizedInputs(CatchEnv(**env_config)))
    config = r2d2.R2D2Config()
    config = config.environment(env_name, env_config=env_config)

    config.lr = 0.0005
    config.gamma = 1
    config.exploration_config['initial_epsilon'] = 1.0
    config.exploration_config['final_epsilon'] = 0.0
    config.exploration_config['epsilon_timesteps'] = 100000 # divide by 1000 to get number of batches
    # config.dueling = False
    # config.hiddens = []

    # from ray.rllib.models import ModelCatalog
    # from rnn import TorchRNNModel, CatchRNNModel
    # ModelCatalog.register_custom_model("my_rnn_model", CatchRNNModel)
    # config.training(model={
    #     'custom_model': 'my_rnn_model',
    #     'custom_model_config': {}
    #     })
    
    config.model['use_lstm'] = True
    config.model['lstm_cell_size'] = 64
    config.model['lstm_use_prev_action'] = True
    config.model['max_seq_len'] = 20
    config.model['fcnet_hiddens'] = [64]
    config.model['fcnet_activation'] = 'linear'
    config.replay_buffer_config['replay_burn_in'] = 20
    config.evaluation_duration = 10; config.evaluation_interval = 1 # evaluate using 10 episodes every episode
    algo = config.build()
    algo.restore(checkpoint)
    return algo, config

def simulate(args):
    env = CatchEnv(render_mode='human', gravity=0, tau=0.015)
    env = NormalizedInputs(env)

    if args.agent == 'model':
        algo, config = load_model(args.checkpoint)

    for _ in range(args.nepisodes):
        obs, info = env.reset()
        if args.agent == 'model':
            state = algo.get_policy().get_initial_state()
            action = 0

        terminated = False
        truncated = False
        while not (terminated or truncated):
            if args.agent == 'human':
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]:
                    action = 2
                elif keys[pygame.K_DOWN]:
                    action = 1
                else:
                    action = 0
            elif args.agent == 'model':
                action, state, q_info = algo.compute_single_action(obs,
                    prev_action=action if config.model['lstm_use_prev_action'] else None,
                    state=state, explore=False)
            elif args.agent == 'random':
                action = env.action_space.sample()
            else:
                action = 2
            obs, reward, terminated, truncated, info = env.step(action)
    env.close()

checkpoint = '/Users/mobeets/ray_results/R2D2_catch_2023-08-15_13-57-463xretgkc/checkpoint_000053'
checkpoint = '/Users/mobeets/ray_results/R2D2_catch_2023-08-15_15-31-59iltng9hq/checkpoint_000007'
checkpoint = '/Users/mobeets/ray_results/R2D2_catch_2023-08-16_10-26-26s182jgc3/checkpoint_000148' # gravity=0
checkpoint = '/Users/mobeets/ray_results/R2D2_catch_2023-08-16_10-54-090n7fr1w6/checkpoint_000013'
checkpoint = '/Users/mobeets/ray_results/R2D2_catch_2023-08-16_11-08-579n2xk7ie/checkpoint_000154'
checkpoint = '/Users/mobeets/ray_results/R2D2_catch_2023-08-16_11-48-3843hz81xd/checkpoint_000186'
checkpoint = '/Users/mobeets/ray_results/R2D2_catch_2023-08-16_12-43-26kqaqwnmv/checkpoint_000003'
checkpoint = '/Users/mobeets/ray_results/R2D2_catch_2023-08-16_12-48-10nk64ocx3/checkpoint_000096'
checkpoint = '/Users/mobeets/ray_results/R2D2_catch_2023-08-16_14-12-388mdkeyso/checkpoint_000096'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, choices=['model', 'human', 'random'],
                        default='random', help='agent choice')
    parser.add_argument('--checkpoint', type=str,
                        default=checkpoint, help='checkpoint directory to load')
    parser.add_argument('-n', '--nepisodes', type=int,
                        default=10, help='number of episodes')
    args = parser.parse_args()
    simulate(args)
