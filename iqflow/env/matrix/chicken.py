"""
Chicken dilemma environment from LOLA at https://github.com/alshedivat/lola/tree/0e8b13e480d6483007549ddf1f660c2f45460fda.
"""
import gym
import numpy as np

from gym.spaces import Discrete, Tuple
from iqflow.env.matrix.util import OneHot


class ChickenGame(gym.Env):
    """
    A two-agent vectorized environment for the Chicken game.
    Possible actions for each agent are (C)ooperate and (D)efect.
    """
    NAME = 'Chicken'
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = 5

    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.payout_mat = np.array([[3., 4.], [1., 0.]])
        self.action_space = \
            Tuple([Discrete(self.NUM_ACTIONS), Discrete(self.NUM_ACTIONS)])
        self.observation_space = \
            Tuple([OneHot(self.NUM_STATES), OneHot(self.NUM_STATES)])

        self.step_count = None

    def reset(self):
        self.step_count = 0
        init_state = np.zeros(self.NUM_STATES)
        init_state[-1] = 1
        observations = [init_state, init_state]
        return observations

    def step(self, action):
        ac0, ac1 = action

        self.step_count += 1

        rewards = np.array([self.payout_mat[ac1][ac0], self.payout_mat[ac0][ac1]])

        state = np.zeros(self.NUM_STATES)
        state[ac0 * 2 + ac1] = 1
        observations = [state, state]

        done = (self.step_count == self.max_steps)

        return observations, rewards, done, 'info'


class Chicken(ChickenGame):

    def __init__(self, config):

        super().__init__(max_steps=config.max_steps)
        self.n_agents = 2
        self.l_action = 2
        self.dim_obs = 5
        self.max_steps = config.max_steps
        self.name = 'chicken'
