import pygame
from tasks.catch import CatchEnv

actor_mode = 'model'
render_mode = 'human'
nepisodes = 10
env = CatchEnv(render_mode=render_mode, gravity=0)

#%% load model, if necessary

checkpoint = '/Users/mobeets/ray_results/R2D2_catch_2023-08-15_13-57-463xretgkc/checkpoint_000053'
checkpoint = '/Users/mobeets/ray_results/R2D2_catch_2023-08-15_15-31-59iltng9hq/checkpoint_000007'
checkpoint = '/Users/mobeets/ray_results/R2D2_catch_2023-08-16_10-26-26s182jgc3/checkpoint_000148' # gravity=0

if actor_mode == 'model':
    from ray.rllib.algorithms import r2d2
    from ray.tune.registry import register_env
    
    env_config = {}
    env_name = 'catch'
    register_env(env_name, lambda env_config: CatchEnv(**env_config))

    config = r2d2.R2D2Config()
    config = config.environment(env_name, env_config=env_config)
    # config.rollouts(num_rollout_workers=0)
    config.lr = 0.0005
    config.gamma = 1
    config.exploration_config['initial_epsilon'] = 1.0
    config.exploration_config['final_epsilon'] = 0.0
    config.exploration_config['epsilon_timesteps'] = 100000 # divide by 1000 to get number of batches
    # config.dueling = False
    # config.hiddens = []
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

#%% simulate

for _ in range(nepisodes):
    obs, info = env.reset()
    if actor_mode == 'model':
        state = algo.get_policy().get_initial_state()
        action = 0

    terminated = False
    truncated = False
    while not (terminated or truncated):
        if actor_mode == 'human':
            action = 2 # default to no action
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action = 1
            elif keys[pygame.K_DOWN]:
                action = 0
            else:
                action = 2
            # events = pygame.event.get()
            # for event in events:
            #     if event.type == pygame.KEYDOWN:
            #         if event.key == pygame.K_UP:
            #             action = 1
            #         if event.key == pygame.K_DOWN:
            #             action = 0
        elif actor_mode == 'model':
            action, state, q_info = algo.compute_single_action(obs,
                prev_action=action if config.model['lstm_use_prev_action'] else None,
                state=state, explore=False)
        else:
            action = env.action_space.sample()
            # action = 0
        obs, reward, terminated, truncated, info = env.step(action)
if render_mode is not None:
    env.close()
