from __future__ import print_function
from scipy import stats

import os
import time
import numpy as np


def test_room_symmetric(n_eval, env, list_agents, mechanism,
                        alg='qflow', log=False, log_path=''):
    rewards_total = np.zeros(env.n_agents)
    n_move_lever = np.zeros(env.n_agents)
    n_move_door= np.zeros(env.n_agents)
    rewards_received= np.zeros(env.n_agents)
    r_lever = np.zeros(env.n_agents)
    r_start = np.zeros(env.n_agents)
    r_door = np.zeros(env.n_agents)
    env_rewards_total = np.zeros(env.n_agents)

    total_steps = 0

    for idx_episode in range(1, n_eval + 1):

        if log:
            given_at_state = np.zeros(3)
            received_at_state = np.zeros(3)
        list_obs = env.reset()
        done = False
        while not done:
            list_actions = []
            for idx, agent in enumerate(list_agents):
                action = agent.run_actor(list_obs[idx])
                list_actions.append(action)
                if action == 0:
                    n_move_lever[idx] += 1
                elif action == 2:
                    n_move_door[idx] += 1

            list_rewards = []

            matrix_given = np.zeros((env.n_agents,))

            mech_reward = mechanism.give_reward(
                    list_obs, list_actions)

            rewards_received += mech_reward
            matrix_given = mech_reward


            for idx, agent in enumerate(list_agents):
                received = matrix_given[idx]
                if list_actions[idx] == 0:
                    r_lever[idx] += received
                elif list_actions[idx] == 1:
                    r_start[idx] += received
                else:
                    r_door[idx] += received
                if log:
                    received_at_state[list_actions[idx]] += matrix_given[idx]

            list_obs_next, env_rewards, done, _ = env.step(list_actions)

            rewards_total += env_rewards
            env_rewards_total += env_rewards

            for idx in range(env.n_agents):
                rewards_total[idx] += matrix_given[idx]

            list_obs = list_obs_next

        total_steps += env.steps

    rewards_total /= n_eval
    env_rewards_total /= n_eval
    n_move_lever /= n_eval
    n_move_door /= n_eval
    rewards_received /= n_eval
    steps_per_episode = total_steps / n_eval
    r_lever /= n_eval
    r_start /= n_eval
    r_door /= n_eval

    return (rewards_total, n_move_lever, n_move_door, rewards_received,
            steps_per_episode, r_lever, r_start, r_door, env_rewards_total.sum())


def test_room_symmetric_base(n_eval, env, list_agents, 
                        alg='pg'):

    rewards_total = np.zeros(env.n_agents)
    n_move_lever = np.zeros(env.n_agents)
    n_move_door= np.zeros(env.n_agents)
    env_rewards_total = np.zeros(env.n_agents)

    total_steps = 0

    for idx_episode in range(1, n_eval + 1):

        list_obs = env.reset()
        done = False
        while not done:
            list_actions = []
            for idx, agent in enumerate(list_agents):
                action = agent.run_actor(list_obs[idx])
                list_actions.append(action)
                if action == 0:
                    n_move_lever[idx] += 1
                elif action == 2:
                    n_move_door[idx] += 1

            list_obs_next, env_rewards, done, _ = env.step(list_actions)

            rewards_total += env_rewards
            env_rewards_total += env_rewards

            list_obs = list_obs_next

        total_steps += env.steps

    rewards_total /= n_eval
    n_move_lever /= n_eval
    n_move_door /= n_eval
    steps_per_episode = total_steps / n_eval
    env_rewards_total /= n_eval


    return (rewards_total, n_move_lever, n_move_door,
            steps_per_episode, env_rewards_total.sum())