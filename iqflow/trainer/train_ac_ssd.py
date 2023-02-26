# Training script for the basic Actor Critic algorithm for Cleanup environment

import json
import os, sys
import random
import time
import importlib
import argparse

import numpy as np
import jax

from copy import deepcopy

import iqflow.config.config_ssd_pg as config_ssd_pg
import iqflow.eval.ssd_evaluate as evaluate
import iqflow.env.ssd.ssd as ssd
import iqflow.env.ssd.ssd_centralized as ssd_centralized

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


    if config.pg.centralized:
        env = ssd_centralized.Env(config.env)
    else:
        env = ssd.Env(config.env)

    if config.pg.use_actor_critic:
        from iqflow.alg.actor_critic import ActorCritic as Alg
    else:
        raise NotImplementedError

    epsilon = config.pg.epsilon_start
    epsilon_step = (
        (epsilon - config.pg.epsilon_end) / config.pg.epsilon_div)


    # ----------------- Initialize agents ---------------- #
    list_agents = []
    for agent_id in range(env.n_agents):
        rng, agent_rng = jax.random.split(rng, 2)
        list_agents.append(
            Alg(env.env.observation_space, config.pg, 
                env.dim_obs, env.l_action,
                config.nn, 'agent_%d' % agent_id,
                agent_id, rng=agent_rng))
    # ------------------------------------------------------- #

    list_agent_meas = []
    list_suffix = ['reward_env',
                   'reward_total', 'waste_cleared',
                   ]
       
    for agent_id in range(1, env.n_agents + 1):
        for suffix in list_suffix:
            list_agent_meas.append('A%d_%s' % (agent_id, suffix))

    header = 'episode,step_train,step,'
    header += ','.join(list_agent_meas)
    header += ',time,reward_env_total\n'
    with open(os.path.join(log_path, 'log.csv'), 'w') as f:
        f.write(header)

    step = 0
    step_train = 0
    t_start = time.time()
    reward_total_return = None
    for idx_episode in range(1, n_episodes + 1):

        list_buffers = run_episode(env, list_agents, epsilon)
        step += len(list_buffers[0].obs)

        for idx, agent in enumerate(list_agents):
            agent.train(list_buffers[idx], epsilon)

        step_train += 1

        if idx_episode % period == 0:

            (reward_env,
            reward_total, waste_cleared) = evaluate.test_ssd_base(
                    n_eval, env, list_agents)
            combined = np.stack([reward_env,
                                 reward_total, waste_cleared])
            s = '%d,%d,%d' % (idx_episode, step_train, step)
            for idx in range(env.n_agents):
                s += ','
                s += '{:.3e},{:.3e},{:.3e}'.format(
                    *combined[:, idx])
            reward_env_total = np.sum(combined[0])
            if reward_total_return is None:
                reward_total_return = reward_total
            else:
                reward_total_return = (reward_total_return * (0.99) +
                 (1 - 0.99) * reward_total)

            s += ',%d,%.2e\n' % (int(time.time()-t_start), reward_env_total)
            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(s)

        if epsilon > config.pg.epsilon_end:
            epsilon -= epsilon_step

    return reward_total_return
    

def run_episode(env, list_agents, epsilon):

    list_buffers = [Buffer(env.n_agents) for _ in range(env.n_agents)]
    list_obs = env.reset()
    done = False

    while not done:
        list_actions = []
        for agent in list_agents:
            action = agent.run_actor(list_obs[agent.agent_id], epsilon)
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

    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str,
                        choices=['er', 'ssd'])
    args = parser.parse_args()    

    config = config_ssd_pg.get_config()

    train_function(config)
