# Config script for IQFlow algorithm for Iterated Matrix Games

from ml_collections import ConfigDict


def get_config():

    config = ConfigDict()

    config.alg = ConfigDict()
    config.alg.n_episodes = 10000
    config.alg.n_eval = 10
    config.alg.n_test = 2
    config.alg.period = 100

    config.env = ConfigDict()
    config.env.name = 'ipd'
    config.env.max_steps = 5
    config.env.n_agents = 2
    config.env.r_multiplier = 2.0  # scale up sigmoid output

    config.qflow = ConfigDict()
    config.qflow.entropy_coeff = 0.05
    config.qflow.gamma = 0.99
    config.qflow.lr_actor = 1e-3
    config.qflow.lr_reward = 3e-3
    config.qflow.lr_v = 1e-3
    config.qflow.lr_v_model = 1e-3
    config.qflow.lr_v_rewarder = 1e-3
    config.qflow.cost_reg_coeff = 0.0
    config.qflow.cost_reg_coeff2 = 0.0

    config.qflow.expectile = 0.85
    config.qflow.use_mini_init = False
    config.qflow.cost_type = 0
    config.qflow.obj_mask = True

    config.qflow.tau = 0.01
    config.qflow.embed_dim = 128
    config.qflow.batch_size = 64
    config.qflow.meta_batch_size = 64
    config.qflow.buffer_size = 1000000
    config.qflow.mech_buffer_size = 1000000

    config.qflow.start_train = 200
    config.qflow.use_actor_critic = False
    config.qflow.period = 1

    config.main = ConfigDict()
    config.main.root_dir_name = 'results/img_results'
    config.main.dir_name = 'ipd_iqflow'
    config.main.exp_name = 'iqflow'
    config.main.max_to_keep = 40
    config.main.model_name = 'model.ckpt'
    config.main.save_period = 30
    config.main.print_period = 10
    config.main.seed = 12341
    config.main.summarize = False
    config.main.report_decay = 0.02

    config.nn = ConfigDict()
    config.nn.n_h1 = 16
    config.nn.n_h2 = 8
    config.nn.n_hr1 = 16
    config.nn.n_hr2 = 8

    return config
