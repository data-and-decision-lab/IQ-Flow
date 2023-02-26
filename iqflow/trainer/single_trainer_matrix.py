# Training script for the IQFlow algorithm for Iterated Matrix Game environment

import ray
import iqflow.env.ssd.ssd as ssd
from copy import deepcopy
from typing import Sequence, Tuple, Optional
from flax.core.frozen_dict import FrozenDict
from flax.training import checkpoints

import json
import os
import random
import sys
import importlib

import numpy as np
import jax
import jax.numpy as jnp
import time

import iqflow.env.ssd.ssd as ssd
import iqflow.config.config_img_qflow as config_img_qflow

import iqflow.eval.ipd_evaluate as evaluate
import iqflow.utils.utils as util
import iqflow.env.matrix.ipd as ipd
import iqflow.env.matrix.chicken as chicken
import iqflow.env.matrix.stag_hunt as stag_hunt

from iqflow.alg.qflow import IQFlow as Mech
from iqflow.utils.utils import Buffer, MechBuffer

from iqflow.networks.common import InfoDict, Model, PRNGKey, Params



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

    with open(os.path.join(log_path, 'config.json'), 'w') as f:
        json.dump(config.to_json(), f, indent=4, sort_keys=True)

    config.main.rng = rng

    n_episodes = config.alg.n_episodes
    n_eval = config.alg.n_eval
    period = config.alg.period

    if config.exp_type == 'ipd':
        env = ipd.IPD(config.env)
    elif config.exp_type == 'chicken':
        env = chicken.Chicken(config.env)
    else:
        env = stag_hunt.StagHunt(config.env)

    from iqflow.alg.policy_gradient import PolicyGradient as Alg

    list_agents = []
    for agent_id in range(env.n_agents):
        rng, agent_rng = jax.random.split(rng, 2)
        list_agents.append(
            Alg(env.observation_space[0], config.qflow, 
                env.dim_obs, env.l_action,
                config.nn, 'agent_%d' % agent_id,
                agent_id, rng=agent_rng))

    list_agent_meas = []
    list_suffix = ['received', 'reward_env',
                       'reward_total', 'n_c', 'n_d']
    for agent_id in range(1, env.n_agents + 1):
        for suffix in list_suffix:
            list_agent_meas.append('A%d_%s' % (agent_id, suffix))

    header = 'episode,step_train,step,'
    header += ','.join(list_agent_meas)
    header += ',time,reward_env_total\n'
    with open(os.path.join(log_path, 'log.csv'), 'w') as f:
        f.write(header)


    list_obs = env.reset()
    list_actions = []
    for agent in list_agents:
        action = agent.run_actor(list_obs[agent.agent_id])
        list_actions.append(action)

    mechanism = Mech(env.observation_space[0], config.qflow, 
                        env.dim_obs, env.l_action,
                        config.nn, 'agent_qflow',
                        config.env.r_multiplier, env.n_agents,
                        rng=rng)

    step = 0
    step_train = 0
    idx_episode = 0
    t_start = time.time()
    prev_reward_env = 0
    reward_env_total_return = None
    while idx_episode < n_episodes:
        
        # print('idx_episode', idx_episode)
        list_buffers, mech_buffer = run_episode(env, list_agents,
                                                mechanism, env.l_action, config.env.r_multiplier, env.l_action)

        mechanism.insert_trajectory(mech_buffer)
        

        if step >= config.qflow.start_train:
            info = mechanism.train()
        else:
            info = mechanism.train_critic()

        step += len(list_buffers[0].obs)
        idx_episode += 1
        
        # Standard learning step for all agents
        for idx, agent in enumerate(list_agents):
            agent.train(list_buffers[idx])

        step_train += 1


        if idx_episode % period == 0:
            received, reward_env, reward_total, n_c, n_d = evaluate.test_mech_ipd(n_eval, env, list_agents, mechanism)

            combined = np.stack([received, reward_env,
                                 reward_total, n_c, n_d])
            s = '%d,%d,%d' % (idx_episode, step_train, step)
            for idx in range(env.n_agents):
                s += ','
                s += '{:.2e},{:.2e},{:.2e},{:.2f},{:.2e}'.format(
                    *combined[:, idx])
            reward_env_total = np.sum(combined[1])
            if reward_env_total_return is None:
                reward_env_total_return = reward_env_total
            else:
                reward_env_total_return = (reward_env_total_return * (1 - config.main.report_decay) +
                 config.main.report_decay * reward_env_total)

            s += ',%d,%.2e\n' % (int(time.time()-t_start), reward_env_total)
            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(s)

            if idx_episode % save_period == 0:
                checkpoints.save_checkpoint(ckpt_dir=log_path+'/checkpoints', target=mechanism.rewarder,
                                            prefix='checkpoint_rewarder_',
                                            overwrite=True,
                                            step=idx_episode, keep=config.main.save_period)
                checkpoints.save_checkpoint(ckpt_dir=log_path+'/checkpoints', target=mechanism.coop_qcritic,
                                            prefix='checkpoint_coopqcritic_',
                                            overwrite=True,
                                            step=idx_episode, keep=config.main.save_period)
                checkpoints.save_checkpoint(ckpt_dir=log_path+'/checkpoints', target=mechanism.env_model_qcritic,
                                            prefix='checkpoint_envmodelqcritic_',
                                            overwrite=True,
                                            step=idx_episode, keep=config.main.save_period)
                checkpoints.save_checkpoint(ckpt_dir=log_path+'/checkpoints', target=mechanism.rewarder_model_qcritic,
                                            prefix='checkpoint_rewardermodelqcritic_',
                                            overwrite=True,
                                            step=idx_episode, keep=config.main.save_period)

    return reward_env_total_return


def run_episode(env, list_agents, mechanism, n_actions_for_r=None, r_multiplier=None, 
                n_actions=None):

    list_buffers = [Buffer(env.n_agents) for _ in range(env.n_agents)]
    mech_buffer = MechBuffer(env.n_agents)
    list_obs = env.reset()
    done = False

    budgets = np.zeros(env.n_agents)

    while not done:

        list_actions = []
        for agent in list_agents:
            action = agent.run_actor(list_obs[agent.agent_id])
            list_actions.append(action)

        mech_rewards = mechanism.give_reward(list_obs, list_actions)

        list_obs_next, env_rewards, done, info = env.step(list_actions)
        budgets += env_rewards

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

    config = config_img_qflow.get_config()
    train_function(config)
