from ray.rllib.algorithms.r2d2 import R2D2Config

def build_agent(latent_size, max_seq_len, use_prev_action=False, use_prev_reward=False):

    config = R2D2Config()
    config = config.framework('torch')
    # config.environment(disable_env_checking=True)
    config.dueling = False
    config.double_q = False
    config.hiddens = []
    config.model['use_lstm'] = True
    config.model['lstm_cell_size'] = latent_size
    config.model['lstm_use_prev_action'] = use_prev_action
    config.model['lstm_use_prev_reward'] = use_prev_reward
    config.model['max_seq_len'] = max_seq_len
    config.model['fcnet_hiddens'] = []
    config.model['fcnet_activation'] = 'linear'
    return config
