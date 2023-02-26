from __future__ import print_function
from scipy import stats
import time, os
import numpy as np

from iqflow.alg import scripted_agents


# Map from name of map to the largest column position
# where a cleaning beam fired from that position can clear waste
cleanup_map_river_boundary = {'cleanup_small_sym': 2,
                              'cleanup_10x10_sym': 3}


def test_mech_ssd(n_eval, env, list_agents, mechanism, alg='mech',
             log=False, log_path='', render=False):

    rewards_env = np.zeros((n_eval, env.n_agents))
    rewards_received = np.zeros((n_eval, env.n_agents))
    rewards_total = np.zeros((n_eval, env.n_agents))
    waste_cleared = np.zeros((n_eval, env.n_agents))
    received_riverside = np.zeros((n_eval, env.n_agents))
    received_beam = np.zeros((n_eval, env.n_agents))
    received_cleared = np.zeros((n_eval, env.n_agents))

    epsilon = 0
    for idx_episode in range(1, n_eval + 1):

        list_obs = env.reset()
        budgets = np.zeros(env.n_agents)
        done = False
        if render:
            env.render()
            input('Episode %d. Press enter to start: ' % idx_episode)

        while not done:
            list_actions = []
            list_binary_actions = []
            for idx, agent in enumerate(list_agents):
                action = agent.run_actor(list_obs[idx], epsilon)
                list_actions.append(action)
                list_binary_actions.append(
                    1 if action == env.cleaning_action_idx else 0)

            # These are the positions seen by the incentive function
            list_agent_positions = env.env.agent_pos

            vector_given = np.zeros((env.n_agents,))

            if env.obs_cleaned_1hot:
                mech_reward = mechanism.give_reward(
                    list_obs, list_binary_actions)
            else:
                mech_reward = mechanism.give_reward(
                    list_obs, list_actions)
                    
            rewards_received[idx_episode-1] += mech_reward
            vector_given = mech_reward

            
            list_obs_next, env_rewards, done, info = env.step(list_actions)
            if render:
                env.render()
                time.sleep(0.1)

            rewards_env[idx_episode-1] += env_rewards
            budgets += env_rewards
            rewards_total[idx_episode-1] += env_rewards

            for idx in range(env.n_agents):
                rewards_total[idx_episode-1, idx] += vector_given[idx]

            waste_cleared[idx_episode-1] += np.array(info['n_cleaned_each_agent'])
            
            for idx in range(env.n_agents):
                received = vector_given[idx]
                if (list_agent_positions[idx][1] <=
                    cleanup_map_river_boundary[env.config.map_name]):
                    received_riverside[idx_episode-1, idx] += received
                if list_binary_actions[idx] == 1:
                    received_beam[idx_episode-1, idx] += received
                if info['n_cleaned_each_agent'][idx] > 0:
                    received_cleared[idx_episode-1, idx] += received

            list_obs = list_obs_next

    rewards_env = np.average(rewards_env, axis=0)
    rewards_received = np.average(rewards_received, axis=0)
    rewards_total = np.average(rewards_total, axis=0)
    waste_cleared = np.average(waste_cleared, axis=0)
    received_riverside = np.average(received_riverside, axis=0)
    received_beam = np.average(received_beam, axis=0)
    received_cleared = np.average(received_cleared, axis=0)

    return (rewards_received, rewards_env,
            rewards_total, waste_cleared, received_riverside,
            received_beam, received_cleared)


def measure_incentive_behavior(env, list_agents, log_path, episode,
                               idx_replace, mechanism=None):

    A1 = scripted_agents.A1(env)
    A2 = scripted_agents.A2(env)
    A3 = scripted_agents.A3(env)
    list_scripted = [A1, A2, A3]
    idx_lio = 1 - idx_replace
    n_eval = 10
    epsilon = 0
    str_write = '%d' % episode
    for idx, scripted_agent in enumerate(list_scripted):

        given = np.zeros(n_eval)

        for idx_episode in range(n_eval):

            list_obs = env.reset()
            done = False
            while not done:
                list_actions = [0, 0]
                list_binary_actions = [0, 0]

                # Run scripted agent
                x_pos = env.env.agent_pos[idx_replace][1]
                action_scripted = scripted_agent.run_actor(x_pos)
                list_actions[idx_replace] = action_scripted
                list_binary_actions[idx_replace] = (
                    1 if action_scripted == env.cleaning_action_idx else 0)

                action_lio = list_agents[idx_lio].run_actor(list_obs[idx_lio], epsilon)
                list_actions[idx_lio] = action_lio
                list_binary_actions[idx_lio] = (
                    1 if action_lio == env.cleaning_action_idx else 0)                

                if mechanism:
                    incentive = mechanism.give_reward(
                        list_obs, list_actions)
                else:
                    raise NotImplementedError

                given[idx_episode] += incentive[idx_replace]  # given to the scripted agent

                list_obs_next, env_rewards, done, info = env.step(list_actions)
                list_obs = list_obs_next

        avg = np.average(given)
        stderr = stats.sem(given)
        str_write += ',%.2e,%.2e' % (avg, stderr)

    str_write += '\n'
    with open(os.path.join(log_path, 'measure_%d.csv'%idx_replace), 'a') as f:
        f.write(str_write)


def test_ssd_base(n_eval, env, list_agents, alg='mech',
             log=False, log_path='', render=False):

    rewards_env = np.zeros((n_eval, env.n_agents))
    rewards_total = np.zeros((n_eval, env.n_agents))
    waste_cleared = np.zeros((n_eval, env.n_agents))

    epsilon = 0
    for idx_episode in range(1, n_eval + 1):

        list_obs = env.reset()
        budgets = np.zeros(env.n_agents)
        done = False
        if render:
            env.render()
            input('Episode %d. Press enter to start: ' % idx_episode)

        while not done:
            list_actions = []
            list_binary_actions = []
            for idx, agent in enumerate(list_agents):
                action = agent.run_actor(list_obs[idx], epsilon)
                list_actions.append(action)
                list_binary_actions.append(
                    1 if action == env.cleaning_action_idx else 0)

            # These are the positions seen by the incentive function
            list_agent_positions = env.env.agent_pos
            
            list_obs_next, env_rewards, done, info = env.step(list_actions)
            if render:
                env.render()
                time.sleep(0.1)

            rewards_env[idx_episode-1] += env_rewards
            budgets += env_rewards
            rewards_total[idx_episode-1] += env_rewards

            waste_cleared[idx_episode-1] += np.array(info['n_cleaned_each_agent'])

            list_obs = list_obs_next


    rewards_env = np.average(rewards_env, axis=0)
    rewards_total = np.average(rewards_total, axis=0)
    waste_cleared = np.average(waste_cleared, axis=0)

    return (rewards_env,
            rewards_total, waste_cleared)