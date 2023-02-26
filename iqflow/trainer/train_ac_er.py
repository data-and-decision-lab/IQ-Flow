# Training script for the ID algorithm for Escape Room environment

import json
import os, sys
import random
import time
import importlib
import argparse

import numpy as np
import jax

from copy import deepcopy

import iqflow.config.config_er_pg as config_er_pg
import iqflow.eval.er_evaluate as evaluate
from iqflow.env.er import room_symmetric

from iqflow.utils.utils import Buffer


def train_function(config):

    seed = config.main.seed
    np.random.seed(seed)
    random.seed(seed)

    dir_name = config.main.dir_name
    exp_name = config.main.exp_name
    root_dir_name = config.main.root_dir_name
    log_path = os.path.join(root_dir_name, exp_name, dir_name)
    model_name = config.main.model_name
    save_period = config.main.save_period

    os.makedirs(log_path, exist_ok=True)

    rng = jax.random.PRNGKey(seed)

    # Keep a record of parameters used for this run
    with open(os.path.join(log_path, 'config.json'), 'w') as f:
        json.dump(config.to_json(), f, indent=4, sort_keys=True)

    config.main.rng = rng

    n_episodes = config.alg.n_episodes
    n_eval = config.alg.n_eval
    period = config.alg.period

    env = room_symmetric.Env(config.env)

    if config.pg.use_actor_critic:
        raise NotImplementedError
    else:
        from iqflow.alg.policy_gradient import PolicyGradient as Alg


    # ----------------- Initialize agents ---------------- #
    list_agents = []
    for agent_id in range(env.n_agents):
        rng, agent_rng = jax.random.split(rng, 2)
        list_agents.append(
            Alg(env.observation_space[0], config.pg, 
                env.dim_obs, env.l_action,
                config.nn, 'agent_%d' % agent_id,
                agent_id, rng=agent_rng))
    # ------------------------------------------------------- #

    list_agent_meas = []
    list_suffix = ['reward_total', 'n_lever', 'n_door']
       
    for agent_id in range(1, env.n_agents + 1):
        for suffix in list_suffix:
            list_agent_meas.append('A%d_%s' % (agent_id, suffix))


    header = 'episode,step_train,step,'
    header += ','.join(list_agent_meas)
    header += ',steps_per_eps,total_env_reward\n'
    with open(os.path.join(log_path, 'log.csv'), 'w') as f:
        f.write(header)

    step = 0
    step_train = 0
    t_start = time.time()
    reward_total_return = None
    for idx_episode in range(1, n_episodes + 1):

        list_buffers = run_episode(env, list_agents)
        step += len(list_buffers[0].obs)

        for idx, agent in enumerate(list_agents):
            agent.train(list_buffers[idx])

        step_train += 1

        if idx_episode % period == 0:

            (reward_total, n_lever, n_door,
                steps_per_episode, env_rewards_total) = evaluate.test_room_symmetric_base(
                    n_eval, env, list_agents)
            combined = np.stack([reward_total, n_lever,
                                    n_door])
            s = '%d,%d,%d' % (idx_episode, step_train, step)
            for idx in range(env.n_agents):
                s += ','
                s += '{:.3e},{:.3e},{:.3e}'.format(
                    *combined[:, idx])

            if reward_total_return is None:
                reward_total_return = reward_total
            else:
                reward_total_return = (reward_total_return * (0.99) +
                 (1 - 0.99) * reward_total)

            s += ',%.2f' % steps_per_episode
            s += ',%.2f\n' % env_rewards_total
            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(s)

    return reward_total_return
    

def run_episode(env, list_agents):

    list_buffers = [Buffer(env.n_agents) for _ in range(env.n_agents)]
    list_obs = env.reset()
    done = False

    while not done:
        list_actions = []
        for agent in list_agents:
            action = agent.run_actor(list_obs[agent.agent_id])
            list_actions.append(action)

        list_obs_next, env_rewards, done, info = env.step(list_actions)

        for idx, buf in enumerate(list_buffers):
            buf.add([list_obs[idx], list_actions[idx],
                     env_rewards[idx],
                     list_obs_next[idx], done])
            buf.add_action_all(list_actions)

        list_action_all = list_actions
            
        list_obs = list_obs_next

    return list_buffers



if __name__ == '__main__':

    config = config_er_pg.get_config()

    train_function(config)
