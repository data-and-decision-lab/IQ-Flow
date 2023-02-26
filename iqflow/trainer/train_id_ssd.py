# Training script for the ID algorithm for Cleanup environment

import json
import os
import random
import time

import numpy as np
import jax

from copy import deepcopy

import iqflow.env.ssd.ssd as ssd
import iqflow.config.config_ssd_id as config_ssd_id
import iqflow.eval.ssd_evaluate as evaluate

from iqflow.utils.utils import Buffer, MechBuffer


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

    # Keep a record of parameters used for this run
    with open(os.path.join(log_path, 'config.json'), 'w') as f:
        json.dump(config.to_json(), f, indent=4, sort_keys=True)

    n_episodes = config.alg.n_episodes
    n_eval = config.alg.n_eval
    period = config.alg.period

    rng = jax.random.PRNGKey(seed)
    config.main.rng = rng

    epsilon = config.id.epsilon_start
    epsilon_step = (
        (epsilon - config.id.epsilon_end) / config.id.epsilon_div)

    env = ssd.Env(config.env)

    from iqflow.alg.actor_critic import ActorCritic as Alg
    from iqflow.alg.incentive_designer_ac import IDAC as Mech

    mechanism = Mech(env.env.observation_space, config.id, 
                env.dim_obs, env.l_action,
                config.nn, 'agent_mech',
                config.env.r_multiplier, env.n_agents,
                env.l_action_for_r, rng=rng)

    list_agents = []
    for agent_id in range(env.n_agents):
        rng, agent_rng = jax.random.split(rng, 2)
        list_agents.append(
            Alg(env.env.observation_space, config.id, 
                env.dim_obs, env.l_action,
                config.nn, 'agent_%d' % agent_id,
                agent_id, rng=agent_rng))

    if config.id.decentralized:
        raise NotImplementedError
    else:
        mechanism.receive_list_of_agents(list_agents)

    list_agent_meas = []
    list_suffix = ['received', 'reward_env',
                   'reward_total', 'waste_cleared',
                   'r_riverside', 'r_beam', 'r_cleared']
    for agent_id in range(1, env.n_agents + 1):
        for suffix in list_suffix:
            list_agent_meas.append('A%d_%s' % (agent_id, suffix))

    header = 'episode,step_train,step,'
    header += ','.join(list_agent_meas)
    header += ',time,reward_env_total\n'
    with open(os.path.join(log_path, 'log.csv'), 'w') as f:
        f.write(header)

    # Log file for measuring incentive behavior w.r.t. 3 scripted agents
    header = 'episode'
    for idx in range(3):
        header += ',A%d_avg,A%d_stderr' % (idx+1, idx+1)
    header += '\n'
    for idx_replace in [0, 1]:
        with open(os.path.join(log_path, 'measure_%d.csv'%idx_replace), 'w') as f:
            f.write(header)

    # Measure behavior at initialization
    if env.n_agents == 2:
        for idx_replace in [0, 1]:
            evaluate.measure_incentive_behavior(
                env, list_agents, log_path, 0, idx_replace, mechanism)

    step = 0
    step_train = 0
    idx_episode = 0
    t_start = time.time()
    prev_reward_env = 0
    reward_env_total_return = None
    list_old_actor_params = None
    mech_buffer_old = None
    while idx_episode < n_episodes:
        
        # print('idx_episode', idx_episode)
        list_buffers, mech_buffer = run_episode(env, list_agents, epsilon,
                                                mechanism)
        step += len(list_buffers[0].obs)
        idx_episode += 1

        if list_old_actor_params:
            mech_buffer_new = deepcopy(mech_buffer)
            mechanism.train_reward(list_old_actor_params,
                                mech_buffer_old, mech_buffer_new, epsilon)
            mech_buffer_old = mech_buffer_new
        else:
            mech_buffer_old = deepcopy(mech_buffer)

        list_old_actor_params = [deepcopy(agent.actor.params) for agent in mechanism.list_of_agents]

        
        # Standard learning step for all agents
        for idx, agent in enumerate(list_agents):
            agent.train(list_buffers[idx], epsilon)

        step_train += 1

        if idx_episode % period == 0:
            (received, reward_env, reward_total,
             waste_cleared, r_riverside, r_beam,
             r_cleared) = evaluate.test_mech_ssd(n_eval, env, list_agents, mechanism)

            combined = np.stack([received, reward_env,
                                 reward_total, waste_cleared,
                                 r_riverside, r_beam, r_cleared])
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
            s += ',%d,%.2e\n' % (int(time.time()-t_start), reward_env_total)
            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(s)

        if env.n_agents == 2 and idx_episode % save_period == 0:
            for idx_replace in [0, 1]:
                evaluate.measure_incentive_behavior(env, list_agents,
                                                    log_path, idx_episode,
                                                    idx_replace,
                                                    mechanism)

        if epsilon > config.id.epsilon_end:
            epsilon -= epsilon_step

    return reward_env_total_return


def run_episode(env, list_agents, epsilon, mech_agent):

    list_buffers = [Buffer(env.n_agents) for _ in range(env.n_agents)]
    mech_buffer = MechBuffer(env.n_agents)
    list_obs = env.reset()
    done = False

    budgets = np.zeros(env.n_agents)

    while not done:
        list_actions = []
        list_binary_actions = []
        for agent in list_agents:
            action = agent.run_actor(list_obs[agent.agent_id],
                                     epsilon)
            list_actions.append(action)
            list_binary_actions.append(1 if action == env.cleaning_action_idx else 0)


        if env.obs_cleaned_1hot:
            mech_rewards = mech_agent.give_reward(list_obs,
                                        list_binary_actions)
        else:
            mech_rewards = mech_agent.give_reward(list_obs,
                                        list_actions)

        list_obs_next, env_rewards, done, info = env.step(list_actions)
        budgets += env_rewards

        for idx, buf in enumerate(list_buffers):
            buf.add([list_obs[idx], list_actions[idx],
                     env_rewards[idx] + mech_rewards[idx],
                     list_obs_next[idx], done])
            if env.obs_cleaned_1hot:
                buf.add_action_all(list_binary_actions)
            else:
                buf.add_action_all(list_actions)

        list_action_all = list_binary_actions if env.obs_cleaned_1hot \
                                             else list_actions

        mech_buffer.add([list_obs, list_actions,
                     env_rewards, mech_rewards,
                     list_obs_next, 
                     np.array([done for _ in range(env.n_agents)]),
                     list_action_all])
            
        list_obs = list_obs_next

    return list_buffers, mech_buffer


if __name__ == '__main__':

    config = config_ssd_id.get_config()
    train_function(config)
