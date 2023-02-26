# Tune the IQFLOW, ID, and AC/PG algorithms for Cleanup, Escape Room, and Iterated Matrix Game environments

import ray
import argparse
import importlib
import os
import multiprocessing as mp

from multiprocessing import Process, Queue
from copy import deepcopy

import numpy as np
import optuna


# mp.set_start_method('forkserver', force=True)


def worker(config, score_queue, train_function):
        score = train_function(config)
        score_queue.put(score)


def main(args, config, train_function, trial_num):

    processes = []
    score_queue = Queue()
    

    n_seeds = args.n_seeds
    seed_min = args.seed_min + trial_num * 995
    seed_base = args.seed_base + trial_num * 995
    dir_name_base = config.main.dir_name

    # Specify the group that contains the variable
    group = 'main'

    # Specify the name of the variable to sweep
    variable = 'seed'

    # Specify the range of values to sweep through
    values = range(n_seeds)

    for idx_run in range(len(values)):
        config_copy = deepcopy(config)
        if variable == 'seed':
            config_copy[group][variable] = seed_base + idx_run
            config_copy.main.dir_name = (
                dir_name_base + '_{}_{:1d}'.format(trial_num, seed_base+idx_run - seed_min))
        else:
            val = values[idx_run]
            if group == 'cleanup_params':
                config_copy['env'][group][variable] = val
            else:
                config_copy[group][variable] = val
            config_copy.main.dir_name = (dir_name_base + '_{:s}'.format(variable) + 
                                        '_{:s}'.format(str(val).replace('.', 'p')))
        config_copy[group]["trial_num"] = trial_num

        
        p = Process(target=worker, args=(config_copy, score_queue, train_function))
        p.start()
        processes.append(p)

    worker_scores = [score_queue.get() for _ in range(len(processes))]

    for p in processes:
        p.join()

    return np.mean(worker_scores).item()


def tune_fn(args, config, train_function, trial):

    config = deepcopy(config)
    if args.alg == 'id' and config.id.decentralized is True:
        config.id.lr_opp = trial.suggest_loguniform("id.lr_opp", low=5e-5, high=1e-3)

    elif args.alg == 'id' and config.id.decentralized is False:
        config.id.entropy_coeff = trial.suggest_loguniform("id.entropy_coeff", low=0.005, high=1.0)
        config.id.lr_actor = trial.suggest_loguniform("id.lr_actor", low=1e-5, high=1e-3)
        config.id.lr_reward = trial.suggest_loguniform("id.lr_reward", low=5e-6, high=1e-4)
        config.id.lr_v = trial.suggest_loguniform("id.lr_v", low=1e-5, high=1e-4)
        config.id.lr_v_model = trial.suggest_loguniform("id.lr_v_model", low=1e-5, high=1e-4)
        config.id.reg_coeff = trial.suggest_loguniform("id.reg_coeff", low=1e-5, high=1e-3)

    elif args.alg == 'qflow' and args.exp == 'er':
        config.qflow.lr_reward = trial.suggest_loguniform("qflow.lr_reward", low=5e-5, high=1e-3)
        config.qflow.cost_reg_coeff = trial.suggest_loguniform("qflow.cost_reg_coeff", low=5e-3, high=5e-1)
        config.qflow.cost_reg_coeff2 = trial.suggest_loguniform("qflow.cost_reg_coeff2", low=1e-4, high=1e-2)
    elif args.alg == 'qflow' and args.exp == 'ssd':
        config.qflow.lr_actor = trial.suggest_loguniform("qflow.lr_actor", low=1e-4, high=1e-3)
        config.qflow.lr_v = trial.suggest_loguniform("qflow.lr_v", low=1e-4, high=1e-3)
        config.qflow.lr_v_model = trial.suggest_loguniform("qflow.lr_v_model", low=5e-5, high=1e-3)
        config.qflow.entropy_coeff = trial.suggest_loguniform("qflow.entropy_coeff", low=1e-2, high=1e-1)
        config.qflow.lr_reward = trial.suggest_loguniform("qflow.lr_reward", low=5e-5, high=1e-3)
        config.qflow.cost_reg_coeff = trial.suggest_loguniform("qflow.cost_reg_coeff", low=5e-3, high=5e-1)
        config.qflow.cost_reg_coeff2 = trial.suggest_loguniform("qflow.cost_reg_coeff2", low=1e-5, high=3e-4)

    return main(args, config, train_function, trial.number)


if __name__ == "__main__":
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, choices=['qflow', 'id'],
                        default='qflow')
    parser.add_argument('--exp', type=str, choices=['er', 'ipd', 'stag_hunt', 'chicken', 'ssd'],
                        default='ssd')
    parser.add_argument('--gpus', type=str,
                        default='0')
    parser.add_argument('--config', type=str, choices=['config_ssd_qflow', 'config_ssd_qflow2',
                                                       'config_ipd_qflow', 'config_er_qflow',
                                                       'config_er_id', 'config_er_id2',
                                                       'config_er_pg', 'config_er_qflow2',
                                                       'config_ssd_id', 'config_ssd_id2',
                                                       'config_ssd_pg'],
                        default='config_ssd_qflow2')
    parser.add_argument('--seed_min', type=int, default=12340)
    parser.add_argument('--seed_base', type=int, default=12340)
    parser.add_argument('--n_seeds', type=int, default=5)
    args = parser.parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "False"
    
    config_file = importlib.import_module('.' + args.config, 'iqflow.config')
    config = config_file.get_config()

    processes = []

    if args.alg == 'qflow':
        if args.exp == 'ssd':
            import iqflow.trainer.single_trainer_ssd as train_ssd
            train_function = train_ssd.train_function
        elif args.exp == 'ipd' or args.exp == 'chicken' or args.exp == 'stag_hunt':
            import iqflow.trainer.single_trainer_matrix as train_matrix
            train_function = train_matrix.train_function
        elif args.exp == 'er':
            import iqflow.trainer.single_trainer_er as train_er
            train_function = train_er.train_function
    elif args.alg == 'id':
        if args.exp == 'ssd':
            import iqflow.trainer.train_id_ssd as train_id_ssd
            train_function = train_id_ssd.train_function
        elif args.exp == 'ipd' or args.exp == 'chicken' or args.exp == 'stag_hunt':
            raise NotImplementedError
        elif args.exp == 'er':
            import iqflow.trainer.train_id_er as train_id_er
            train_function = train_id_er.train_function

    elif args.alg == 'pg':
        if args.exp == 'ssd':
            import iqflow.trainer.train_ac_ssd as train_ac_ssd
            train_function = train_ac_ssd.train_function
        elif args.exp == 'ipd' or args.exp == 'chicken' or args.exp == 'stag_hunt':
            raise NotImplementedError
        elif args.exp == 'er':
            import iqflow.trainer.train_ac_er as train_ac_er
            train_function = train_ac_er.train_function

    main_dir = os.path.join(config.main.root_dir_name, config.main.exp_name, config.main.dir_name, 'results_optuna')
    study_name = "tune_" + args.alg
    n_trials = 40
    n_startup_trials = 5

    os.makedirs(main_dir, exist_ok=True)
    storage_url = "".join(("sqlite:///", os.path.join(main_dir, "store.db")))
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
    study = optuna.create_study(
            storage=storage_url,
            sampler=sampler,
            study_name=study_name,
            direction="maximize",
            load_if_exists=True)
    study.optimize(
            lambda trial: tune_fn(args, config, train_function, trial),
            n_trials=n_trials,
            n_jobs=1)