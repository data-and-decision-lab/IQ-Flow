# Config script for Policy Gradient algorithm for Escape Room

from ml_collections import ConfigDict


def get_config():

    config = ConfigDict()

    config.alg = ConfigDict()
    config.alg.n_episodes = 20000
    config.alg.n_eval = 10
    config.alg.period = 100

    config.env = ConfigDict()
    config.env.max_steps = 5
    config.env.min_at_lever = 2
    config.env.n_agents = 5
    config.env.name = 'er'
    config.env.r_multiplier = 2.0
    config.env.randomize = False
    config.env.reward_sanity_check = False
    
    config.pg = ConfigDict()
    config.pg.entropy_coeff = 0.0166
    config.pg.gamma = 0.99
    config.pg.lr_actor = 9.56e-5
    config.pg.use_actor_critic = False

    config.main = ConfigDict()
    config.main.root_dir_name = 'results/er_results'
    config.main.dir_name = 'tune_er_n5_2'
    config.main.exp_name = 'er_pg'
    config.main.max_to_keep = 100
    config.main.model_name = 'model.ckpt'
    config.main.save_period = 100000
    config.main.seed = 12340
    config.main.summarize = False
    config.main.use_gpu = False

    config.nn = ConfigDict()
    config.nn.n_h1 = 64
    config.nn.n_h2 = 64
    config.nn.n_hr1 = 64
    config.nn.n_hr2 = 32

    return config
