# Training script for the ID algorithm for Escape Room environment

import json
import os, sys
import random
import time
import importlib

import numpy as np
import jax

from copy import deepcopy

import iqflow.config.config_er_id as config_er_id
import iqflow.eval.er_evaluate as evaluate
from iqflow.env.er import room_symmetric

from iqflow.utils.utils import Buffer, MechBuffer


def train_function(config):

    dir_name = config.main.dir_name
    exp_name = config.main.exp_name
    root_dir_name = config.main.root_dir_name
    log_path = os.path.join(root_dir_name, exp_name, dir_name)
    model_name = config.main.model_name
    save_period = config.main.save_period

    os.makedirs(log_path, exist_ok=True)

    seed = config.main.seed

    np.random.seed(seed)
    random.seed(seed)

    rng = jax.random.PRNGKey(seed)

    # Keep a record of parameters used for this run
    with open(os.path.join(log_path, 'config.json'), 'w') as f:
        json.dump(config.to_json(), f, indent=4, sort_keys=True)

    config.main.rng = rng

    n_episodes = config.alg.n_episodes
    n_eval = config.alg.n_eval
    period = config.alg.period

    env = room_symmetric.Env(config.env)

    if config.id.use_actor_critic:
        raise NotImplementedError
    else:
        from iqflow.alg.policy_gradient import PolicyGradient as Alg
        from iqflow.alg.incentive_designer_pg import IDPG as Mech

    mech_agent = Mech(env.observation_space[0], config.id, 
                env.dim_obs, env.l_action,
                config.nn, 'agent_mech',
                config.env.r_multiplier, env.n_agents,
                None, rng=rng)

    list_agents = []
    for agent_id in range(env.n_agents):
        rng, agent_rng = jax.random.split(rng, 2)
        list_agents.append(
            Alg(env.observation_space[0], config.id, 
                env.dim_obs, env.l_action,
                config.nn, 'agent_%d' % agent_id,
                agent_id, rng=agent_rng))

    if config.id.decentralized:
        mech_agent.model_agents()
    else:
        mech_agent.receive_list_of_agents(list_agents)

    list_agent_meas = []
    list_suffix = ['reward_total', 'n_lever', 'n_door',
                       'received', 'r-lever', 'r-start', 'r-door']
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
    idx_episode = 0
    t_start = time.time()
    prev_reward_env = 0
    reward_env_total_return = None
    list_old_actor_params = None
    mech_buffer_old = None
    while idx_episode < n_episodes:
        
        list_buffers, mech_buffer = run_episode(env, list_agents,
                                                mech_agent)

        step += len(list_buffers[0].obs)
        idx_episode += 1

        if list_old_actor_params:
            mech_buffer_new = deepcopy(mech_buffer)
            mech_agent.train_reward(list_old_actor_params,
                                mech_buffer_old, mech_buffer_new)
            mech_buffer_old = mech_buffer_new
        else:
            mech_buffer_old = deepcopy(mech_buffer)

        list_old_actor_params = [deepcopy(agent.actor.params) for agent in mech_agent.list_of_agents]

        
        if config.id.decentralized:
            mech_agent.train_agent_models(mech_buffer)
        
        # Standard learning step for all agents
        for idx, agent in enumerate(list_agents):
            agent.train(list_buffers[idx])

        step_train += 1

        if idx_episode % period == 0:
            (reward_total, n_move_lever, n_move_door,
                     rewards_received,
                     steps_per_episode, r_lever,
                     r_start, r_door, env_rewards_total) = evaluate.test_room_symmetric(
                         n_eval, env, list_agents, mech_agent, alg='qflow')
            combined = np.stack([reward_total, n_move_lever, n_move_door,
                                            rewards_received,
                                            r_lever, r_start, r_door])
            s = '%d,%d,%d' % (idx_episode, step_train, step)
            for idx in range(env.n_agents):
                s += ','
                s += '{:.2e},{:.2e},{:.2e},{:.2f},{:.2e},{:.2e},{:.2e}'.format(
                    *combined[:, idx])
            reward_env_total = np.sum(combined[1])
            if reward_env_total_return is None:
                reward_env_total_return = reward_env_total
            else:
                reward_env_total_return = (reward_env_total_return * (0.99) +
                 (1 - 0.99) * reward_env_total)

            s += ',%.2f' % steps_per_episode
            s += ',%.2f\n' % env_rewards_total
            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(s)

    return reward_env_total_return


def run_episode(env, list_agents,  mech_agent):

    list_buffers = [Buffer(env.n_agents) for _ in range(env.n_agents)]
    mech_buffer = MechBuffer(env.n_agents)
    list_obs = env.reset()
    done = False

    while not done:
        list_actions = []
        for agent in list_agents:
            action = agent.run_actor(list_obs[agent.agent_id])
            list_actions.append(action)

        mech_rewards = mech_agent.give_reward(list_obs,
                                    list_actions)

        list_obs_next, env_rewards, done, info = env.step(list_actions)

        for idx, buf in enumerate(list_buffers):
            buf.add([list_obs[idx], list_actions[idx],
                     env_rewards[idx] + mech_rewards[idx],
                     list_obs_next[idx], done])
            buf.add_action_all(list_actions)

        list_action_all = list_actions

        mech_buffer.add([list_obs, list_actions,
                     env_rewards, mech_rewards,
                     list_obs_next, 
                     np.array([done for _ in range(env.n_agents)]),
                     list_action_all])
            
        list_obs = list_obs_next

    return list_buffers, mech_buffer


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    config_file = importlib.import_module('.' + sys.argv[2], 'iqflow.config')

    config = config_er_id.get_config()
    train_function(config)
