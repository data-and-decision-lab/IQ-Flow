from os import replace
import gym
import numpy as np
import jax
import jax.numpy as jnp
import collections


Batch = collections.namedtuple(
    'Batch',
    ['obs', 'action', 'reward', 'obs_next', 'done', 'action_all', 'next_action'])

Queues = collections.namedtuple(
    'Queues',
    ['data', 'rewarder_net', 'start', 'termin', 'cont'])


def make_onehot(index: int, max_size: int):
    return jax.ops.index_add(jnp.zeros(shape=(max_size,)), index, 1)


@jax.partial(jax.jit, static_argnums=1)
def process_actions(index: jnp.array, max_size: int):
    return jax.vmap(make_onehot, (0, None), 0)(index, max_size)

@jax.partial(jax.jit, static_argnums=(1,))
def get_action_1hot_flat(action_all, l_action):
    action_all = jnp.array(action_all)
    actions_1hot = process_actions(action_all, l_action)
    return actions_1hot.flatten()


@jax.partial(jax.jit, static_argnums=(1,))
def get_action_1hot(action_all, l_action):
    action_all = jnp.array(action_all)
    actions_1hot = process_actions(action_all, l_action)
    return actions_1hot


def get_action_others_1hot(action_all, agent_id, l_action):
    action_all = jnp.array([action for idx, action in enumerate(action_all) if idx != agent_id])
    actions_1hot = process_actions(action_all, l_action)
    return actions_1hot.flatten()


def get_action_1hot_flat_batch(list_action_all, l_action):
    actions_1hot = jax.vmap(get_action_1hot_flat, (0, None), 0)(list_action_all, l_action)
    return actions_1hot


def get_action_1hot_batch(list_action_all, l_action):
    actions_1hot = jax.vmap(get_action_1hot, (0, None), 0)(list_action_all, l_action)
    return actions_1hot


def get_action_others_1hot_batch(list_action_all, agent_id, l_action):
    actions_1hot = jax.vmap(get_action_others_1hot, (0, None, None), 0)(list_action_all, agent_id, l_action)
    return actions_1hot


def process_rewards(rewards, gamma):
    n_steps = len(rewards)
    gamma_prod = jnp.cumprod(jnp.ones(n_steps) * gamma)
    returns = jnp.cumsum((rewards * gamma_prod)[::-1])[::-1]
    returns = returns / gamma_prod

    return returns


def asymmetric_loss(diff, expectile=0.85):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class Buffer(object):

    def __init__(self, n_agents): 
        self.n_agents = n_agents
        self.reset()

    def reset(self):
        self.obs = []
        self.obs_v = []
        self.action = []
        self.action_all = []
        self.r_sampled = []
        self.reward = []
        self.obs_next = []
        self.obs_v_next = []
        self.done = []

    def add(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.obs_next.append(transition[3])
        self.done.append(transition[4])
        
    def add_r_sampled(self, r_sampled):
        self.r_sampled.append(r_sampled)

    def add_action_all(self, list_actions):
        self.action_all.append(list_actions)

    def add_obs_v(self, obs_v, obs_v_next):
        self.obs_v.append(obs_v)
        self.obs_v_next.append(obs_v_next)


class MechBuffer(object):

    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.reset()

    def reset(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.obs_next = []
        self.done = []
        self.r_from_mech = []
        self.action_all = []

    def add(self, transition_list):
        self.obs.append(transition_list[0])
        self.action.append(transition_list[1])
        self.reward.append(transition_list[2])
        self.r_from_mech.append(transition_list[3])
        self.obs_next.append(transition_list[4])
        self.done.append(transition_list[5])
        self.action_all.append(transition_list[6])


class MechReplay(object):

    def __init__(self, obs_space: gym.spaces.Box, action_dim: int,
                 n_agents: int, batch_size: int, capacity: int):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.capacity = capacity
        self.batch = collections.namedtuple('Batch',
                                ['obs', 'action', 'reward', 'obs_next', 'done',
                                 'action_all', 'action_1hot_flat',
                                 'action_1hot',
                                 'r_from_mech'])
        self.reset()
        self.size = 0
        self.insert_index = 0

    def reset(self):
        self.obs = np.empty((self.capacity, self.n_agents,
                              *self.obs_space.shape), dtype=self.obs_space.dtype)
        self.action = np.empty((self.capacity, self.n_agents,), dtype=np.int32)
        self.reward = np.empty((self.capacity, self.n_agents,), dtype=np.float32)
        self.obs_next = np.empty((self.capacity, self.n_agents,
                              *self.obs_space.shape), dtype=self.obs_space.dtype)
        self.done = np.empty((self.capacity, self.n_agents,), dtype=np.float32)
        self.r_from_mech = np.empty((self.capacity, self.n_agents,), dtype=np.float32)
        self.action_all = np.empty((self.capacity, self.n_agents,), dtype=np.int32)
        self.action_1hot_flat = np.empty((self.capacity, self.n_agents * self.action_dim,), dtype=np.int32)
        self.action_1hot = np.empty((self.capacity,self.n_agents, self.action_dim, ), dtype=np.int32)

    def insert(self, obs, action, reward, obs_next, done, action_all,
                action_1hot_flat, action_1hot, r_from_mech):
        added_size = len(obs)
        self.obs[self.insert_index:self.insert_index+added_size] = np.array(obs)
        self.action[self.insert_index:self.insert_index+added_size] = np.array(action)
        self.reward[self.insert_index:self.insert_index+added_size] = np.array(reward)
        self.obs_next[self.insert_index:self.insert_index+added_size] = np.array(obs_next)
        self.done[self.insert_index:self.insert_index+added_size] = np.array(done)
        self.action_all[self.insert_index:self.insert_index+added_size] = np.array(action_all)
        self.action_1hot_flat[self.insert_index:self.insert_index+added_size] = np.array(action_1hot_flat)
        self.action_1hot[self.insert_index:self.insert_index+added_size] = np.array(action_1hot)
        self.r_from_mech[self.insert_index:self.insert_index+added_size] = np.array(r_from_mech)

        self.insert_index = (self.insert_index + added_size) % self.capacity
        self.size = min(self.size + added_size, self.capacity)

    def sample(self, batch_size: int = None):
        if batch_size is None:
            batch_size = self.batch_size
        batch_size = min(self.size, batch_size)
        idx = np.random.choice(self.size, size=batch_size, replace=False)
        return self.batch(jnp.array(self.obs[idx]),
                          jnp.array(self.action[idx]),
                          jnp.array(self.reward[idx]),
                          jnp.array(self.obs_next[idx]),
                          jnp.array(self.done[idx]),
                          jnp.array(self.action_all[idx]),
                          jnp.array(self.action_1hot_flat[idx]),
                          jnp.array(self.action_1hot[idx]),
                          jnp.array(self.r_from_mech[idx]))

    def meta_sample(self, batch_size: int = None):
        if batch_size is None:
            batch_size = self.batch_size
        idx = np.random.choice(self.size, size=2 * batch_size, replace=False)
        return self.batch(jnp.array(self.obs[idx[:batch_size]]),
                          jnp.array(self.action[idx[:batch_size]]),
                          jnp.array(self.reward[idx[:batch_size]]),
                          jnp.array(self.obs_next[idx[:batch_size]]),
                          jnp.array(self.done[idx[:batch_size]]),
                          jnp.array(self.action_all[idx[:batch_size]]),
                          jnp.array(self.action_1hot_flat[idx[:batch_size]]),
                          jnp.array(self.action_1hot[idx[:batch_size]]),
                          jnp.array(self.r_from_mech[idx[:batch_size]])), \
                              self.batch(jnp.array(self.obs[idx[batch_size:]]),
                          jnp.array(self.action[idx[batch_size:]]),
                          jnp.array(self.reward[idx[batch_size:]]),
                          jnp.array(self.obs_next[idx[batch_size:]]),
                          jnp.array(self.done[idx[batch_size:]]),
                          jnp.array(self.action_all[idx[batch_size:]]),
                          jnp.array(self.action_1hot_flat[idx[batch_size:]]),
                          jnp.array(self.action_1hot[idx[batch_size:]]),
                          jnp.array(self.r_from_mech[idx[batch_size:]]))
