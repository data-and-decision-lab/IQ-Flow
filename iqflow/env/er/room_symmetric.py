import numpy as np
from iqflow.env.er import room_agent
from iqflow.env.matrix.util import OneHot


class Env(object):

    def __init__(self, config_env):

        self.config = config_env

        self.n_agents = self.config.n_agents
        self.name = 'er'
        self.l_action = 3
        # Observe self position (1-hot),
        # other agents' positions (1-hot for each other agent)
        # total amount given to each other agent
        self.l_obs = 3 * self.n_agents
        self.dim_obs = self.l_obs

        self.max_steps = self.config.max_steps
        self.min_at_lever = self.config.min_at_lever
        self.randomize = self.config.randomize

        self.actors = [room_agent.Actor(idx, self.n_agents, self.l_obs)
                       for idx in range(self.n_agents)]

        self.observation_space = [OneHot(self.l_obs)] * self.n_agents

    def get_door_status(self, actions):
        n_going_to_lever = actions.count(0)
        return n_going_to_lever >= self.min_at_lever

    def calc_reward(self, actions, door_open):
        assert len(actions) == self.n_agents
        rewards = np.zeros(self.n_agents)

        for agent_id in range(0, self.n_agents):
            if door_open and actions[agent_id] == 2:
                rewards[agent_id] = 10
            elif actions[agent_id] == self.actors[agent_id].position:
                # no penalty for staying at current position
                rewards[agent_id] = 0
            else:
                rewards[agent_id] = -1

        return rewards

    def get_obs(self):
        list_obs = []
        for actor in self.actors:
            list_obs.append(actor.get_obs(self.state))

        return list_obs

    def step(self, actions):

        door_open = self.get_door_status(actions)
        rewards = self.calc_reward(actions, door_open)
        for idx, actor in enumerate(self.actors):
            actor.act(actions[idx])
        self.steps += 1
        self.state = [actor.position for actor in self.actors]
        list_obs_next = self.get_obs()

        # Terminate if (door is open and some agent ended up at door)
        # or reach max_steps
        done = (door_open and 2 in self.state) or self.steps == self.max_steps

        return list_obs_next, rewards, done, '_'

    def reset(self):
        for actor in self.actors:
            actor.reset(self.randomize)
        self.state = [actor.position for actor in self.actors]
        self.steps = 0
        list_obs = self.get_obs()

        return list_obs
