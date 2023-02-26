import time
import numpy as np


def test_mech_ipd(n_eval, env, list_agents, mechanism, alg='mech',
             log=False, log_path='', render=False, give_reward=None, n_actions_for_r=None, r_multiplier=None, 
                n_actions=None):

    n_c = np.zeros((n_eval, env.n_agents))  # count of cooperation
    n_d = np.zeros((n_eval, env.n_agents))  # count of defection
    rewards_env = np.zeros((n_eval, env.n_agents))
    rewards_given = np.zeros((n_eval, env.n_agents))
    rewards_received = np.zeros((n_eval, env.n_agents))
    rewards_total = np.zeros((n_eval, env.n_agents))

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
            for idx, agent in enumerate(list_agents):
                action = agent.run_actor(list_obs[idx], epsilon)
                list_actions.append(action)
                if action == 0:
                    n_c[idx_episode-1, idx] += 1
                elif action == 1:
                    n_d[idx_episode-1, idx] += 1

            vector_given = np.zeros((env.n_agents,))
            if give_reward:
                mech_reward = give_reward(mechanism,
                        list_obs, list_actions, n_actions_for_r, r_multiplier, 
                n_actions)
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

            list_obs = list_obs_next

    rewards_env = np.average(rewards_env, axis=0) / env.max_steps
    rewards_given = np.average(rewards_given, axis=0) / env.max_steps
    rewards_received = np.average(rewards_received, axis=0) / env.max_steps
    rewards_total = np.average(rewards_total, axis=0) / env.max_steps
    n_c = np.average(n_c, axis=0) / env.max_steps
    n_d = np.average(n_d, axis=0) / env.max_steps

    return (rewards_received, rewards_env,
            rewards_total, n_c, n_d)