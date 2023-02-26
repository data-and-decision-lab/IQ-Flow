# Config script for ID algorithm for 10x10 Cleanup

from ml_collections import ConfigDict


def get_config():

    config = ConfigDict()

    config.alg = ConfigDict()
    config.alg.n_episodes = 5000
    config.alg.n_eval = 10
    config.alg.n_test = 3
    config.alg.period = 100

    config.env = ConfigDict()
    config.env.beam_width = 3  # default 3
    config.env.cleaning_penalty = 0.0
    config.env.disable_left_right_action = False
    config.env.disable_rotation_action = True
    # if not None, a fixed global reference frame is used for all agents
    # config.env.global_ref_point = [4, 4]  # cleanup_10x10
    # config.env.global_ref_point = [3, 3]  # for cleanup_small
    config.env.global_ref_point = None
    config.env.map_name = 'cleanup_10x10_sym'  # 'cleanup_small_sym'|'cleanup_10x10_sym'
    config.env.max_steps = 50 # small: 50 | 10x10: 50
    config.env.n_agents = 2
    # If T, reward function takes in 1-hot representation of
    # whether the other agent used the cleaning beam
    # Else, observe the full 1-hot action of other agent
    config.env.obs_cleaned_1hot = False
    # ---------- For 10x10 map cleanup_10x10_sym ----------
    config.env.obs_height = 15
    config.env.obs_width = 15
    # -----------------------------------
    # ---------- For 7x7 map cleanup_small_sym ------------
    # config.env.obs_height = 9
    # config.env.obs_width = 9
    # -----------------------------------
    config.env.r_multiplier = 2.0  # scale up sigmoid output
    config.env.random_orientation = False
    config.env.shuffle_spawn = False
    # 0.5(height - 1)
    # config.env.view_size = 4
    config.env.view_size = 7
    config.env.cleanup_params = ConfigDict()
    config.env.cleanup_params.appleRespawnProbability = 0.3  # 10x10 0.3 | small 0.5
    config.env.cleanup_params.thresholdDepletion = 0.4  # 10x10 0.4 | small 0.6
    config.env.cleanup_params.thresholdRestoration = 0.0  # 10x10 0.0 | small 0.0
    config.env.cleanup_params.wasteSpawnProbability = 0.5  # 10x10 0.5 | small 0.5

    config.id = ConfigDict()
    config.id.decentralized = False
    config.id.entropy_coeff = 0.1
    config.id.epsilon_div = 1e3  # small 1e2, 10x10 1e3
    config.id.epsilon_end = 0.05
    config.id.epsilon_start = 1.0
    config.id.gamma = 0.99
    config.id.lr_actor = 6.73e-4
    config.id.lr_reward = 1.24e-5
    config.id.lr_v = 5.54e-4
    config.id.lr_v_model = 2.4e-5
    config.id.optimizer = 'adam'
    config.id.reg_coeff = 1e-4  # float, or 'linear', or 'adaptive'
    config.id.tau = 0.01

    config.main = ConfigDict()
    config.main.root_dir_name = 'results/cleanup_results'
    config.main.dir_name = 'cleanup_big_id'
    config.main.exp_name = 'cleanup_id_ac_tune'
    config.main.max_to_keep = 12
    config.main.model_name = 'model.ckpt'
    config.main.save_period = 1000
    config.main.save_threshold = 40
    config.main.seed = 12340
    config.main.summarize = False
    config.main.report_decay = 0.02

    config.nn = ConfigDict()
    config.nn.kernel = (3, 3)
    config.nn.n_filters = 6
    config.nn.n_h1 = 64
    config.nn.n_h2 = 64
    config.nn.n_h = 128
    config.nn.stride = (1, 1)

    return config
