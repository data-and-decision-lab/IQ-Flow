# Config script for ID algorithm for Escape Room (5, 2)

from ml_collections import ConfigDict


def get_config():

    config = ConfigDict()

    config.alg = ConfigDict()
    config.alg.n_episodes = 20000
    config.alg.n_eval = 10
    config.alg.n_test = 100
    config.alg.name = 'id'
    config.alg.period = 100

    config.env = ConfigDict()
    config.env.max_steps = 5
    config.env.min_at_lever = 2
    config.env.n_agents = 5
    config.env.name = 'er'
    config.env.r_multiplier = 2.0
    config.env.randomize = False

    config.id = ConfigDict()
    config.id.decentralized = False
    config.id.entropy_coeff = 0.0166
    config.id.epsilon_div = 1000
    config.id.epsilon_end = 0.1
    config.id.epsilon_start = 0.5
    config.id.gamma = 0.99
    config.id.lr_actor = 9.56e-5
    config.id.lr_cost = 6.03e-5
    config.id.lr_opp = 1e-3
    config.id.lr_reward = 7.93e-4
    config.id.optimizer = 'adam'
    config.id.reg = 'l1'
    config.id.reg_coeff = 1.0
    config.id.separate_cost_optimizer = True
    config.id.use_actor_critic = False

    config.main = ConfigDict()
    config.main.root_dir_name = 'results/er_results'
    config.main.dir_name = 'small_er_n5_2_id'
    config.main.exp_name = 'er_id_pg_tune'
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
                            
                
