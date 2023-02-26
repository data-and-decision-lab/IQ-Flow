# Config script for IQFlow algorithm for Escape Room (5, 2)

from ml_collections import ConfigDict


def get_config():

    config = ConfigDict()

    config.alg = ConfigDict()
    config.alg.n_episodes = 25000
    config.alg.n_eval = 10
    config.alg.n_test = 100
    config.alg.name = 'qflow'
    config.alg.period = 100

    config.env = ConfigDict()
    config.env.max_steps = 5
    config.env.min_at_lever = 2
    config.env.n_agents = 5
    config.env.name = 'er'
    config.env.r_multiplier = 2.0
    config.env.randomize = False

    config.qflow = ConfigDict()
    config.qflow.entropy_coeff = 0.0166
    config.qflow.epsilon_div = 1000
    config.qflow.epsilon_end = 0.1
    config.qflow.epsilon_start = 1.0
    config.qflow.gamma = 0.99
    config.qflow.lr_actor = 9.56e-5
    config.qflow.lr_reward = 1e-3
    config.qflow.lr_v_model = 1e-3
    config.qflow.lr_v_rewarder = 1e-3
    config.qflow.cost_reg_coeff = 5e-1
    config.qflow.cost_reg_coeff2 = 5e-3

    config.qflow.expectile = 0.85
    config.qflow.use_mini_init = False
    config.qflow.cost_type = 0
    config.qflow.obj_mask = True

    config.qflow.tau = 0.01
    config.qflow.embed_dim = 128
    config.qflow.batch_size = 64
    config.qflow.meta_batch_size = 64
    config.qflow.buffer_size = 1000000
    config.qflow.start_train = 300
    config.qflow.use_actor_critic = False

    config.main = ConfigDict()
    config.main.root_dir_name = 'results/er_results'
    config.main.dir_name = 'small_n5_2'
    config.main.exp_name = 'er_qflow_pg'
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
                            
                
