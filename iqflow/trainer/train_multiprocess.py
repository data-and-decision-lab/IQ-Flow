# Train the IQFLOW, ID, and AC/PG algorithms with multiprocessing for Cleanup, Escape Room, and Iterated Matrix Game environments

import ray
import argparse
import importlib
import os
import multiprocessing as mp

from multiprocessing import Process
from multiprocessing import Manager
from copy import deepcopy

mp.set_start_method('forkserver', force=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, choices=['qflow', 'id', 'pg'],
                        default='qflow')
    parser.add_argument('--exp', type=str, choices=['er', 'ipd', 'stag_hunt', 'chicken', 'ssd'],
                        default='ssd')
    parser.add_argument('--gpus', type=str,
                        default='0')
    parser.add_argument('--config', type=str, choices=['config_ssd_qflow', 'config_ssd_qflow2',
                                                       'config_img_qflow', 'config_er_qflow',
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
            import single_trainer_ssd as train_ssd
            train_function = train_ssd.train_function
        elif args.exp == 'ipd' or args.exp == 'chicken' or args.exp == 'stag_hunt':
            import single_trainer_matrix as train_matrix
            train_function = train_matrix.train_function
        elif args.exp == 'er':
            import single_trainer_er as train_er
            train_function = train_er.train_function
    elif args.alg == 'id':
        if args.exp == 'ssd':
            import train_id_ssd
            train_function = train_id_ssd.train_function
        elif args.exp == 'ipd' or args.exp == 'chicken' or args.exp == 'stag_hunt':
            raise NotImplementedError
        elif args.exp == 'er':
            import train_id_er
            train_function = train_id_er.train_function

    elif args.alg == 'pg':
        if args.exp == 'ssd':
            import train_ac_ssd
            train_function = train_ac_ssd.train_function
        elif args.exp == 'ipd' or args.exp == 'chicken' or args.exp == 'stag_hunt':
            raise NotImplementedError
        elif args.exp == 'er':
            import train_ac_er
            train_function = train_ac_er.train_function

    n_seeds = args.n_seeds
    seed_min = args.seed_min
    seed_base = args.seed_base
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
                dir_name_base + '_{:1d}'.format(seed_base+idx_run - seed_min))
        else:
            val = values[idx_run]
            if group == 'cleanup_params':
                config_copy['env'][group][variable] = val
            else:
                config_copy[group][variable] = val
            config_copy.main.dir_name = (dir_name_base + '_{:s}'.format(variable) + 
                                        '_{:s}'.format(str(val).replace('.', 'p')))

        config_copy['exp_type'] = args.exp
        p = Process(target=train_function, args=(config_copy,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
