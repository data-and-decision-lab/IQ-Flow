# Actor Critic based Incentive Designer (ID) algorithm

from random import seed
from typing import Sequence, Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
import flax.linen as nn
import collections

from gym.spaces import MultiDiscrete

from iqflow.networks import nets

from iqflow.utils import utils as util

from iqflow.networks.common import InfoDict, Model, PRNGKey, Params


Batch = collections.namedtuple(
    'Batch',
    ['obs', 'action', 'reward', 'obs_next', 'done',
     'r_from_mech', 'action_all'])


@jax.partial(jax.jit, static_argnums=(3, 4))
def _update_jit(
    critic: Model, target_critic: Model, batch: Batch,
    discount: float, tau: float) -> \
    Tuple[Model, Model, InfoDict]:

    new_critic, critic_info = update_critic(critic, target_critic,
                                            batch, discount)
    new_target_critic = target_update(new_critic, target_critic, tau)

    return new_critic, new_target_critic, {
        **critic_info,
    }


class IDAC(object):

    def __init__(self, obs_space, config, dim_obs, n_actions, nn_config, agent_name,
                 r_multiplier, n_agents, n_actions_for_r, rng):
        self.alg_name = 'ID'
        self.dim_obs = dim_obs
        self.image_obs = isinstance(self.dim_obs, list)
        self.n_actions = n_actions
        self.nn = nn_config
        self.agent_name = agent_name
        self.r_multiplier = r_multiplier
        self.n_agents = n_agents
        self.n_actions_for_r = n_actions_for_r if n_actions_for_r else n_actions
        self.obs_space = obs_space


        self.entropy_coeff = config.entropy_coeff
        self.gamma = config.gamma
        self.lr_reward = config.lr_reward
        self.lr_v = config.lr_v
        self.lr_v_model = config.lr_v_model
        self.reg_coeff = config.reg_coeff
        self.tau = config.tau
        self.epsilon_start = config.epsilon_start
        self.config = config

        self.rng = rng

        self.create_networks()
        
    def create_networks(self):
        rng, critic_key, rewarder_key = jax.random.split(self.rng, 3)

        # Observation is either 1D or 3D
        if self.image_obs:
            critic_def = nets.CentralizedVNet(self.nn.n_filters * self.n_agents,
                                           self.nn.kernel, self.nn.stride,
                                           self.nn.n_h1, self.nn.n_h2,
                                           self.n_agents, self.n_actions)
            rewarder_def = nets.Reward(self.nn.n_filters * self.n_agents,
                                       self.nn.kernel,
                                       self.nn.stride, self.nn.n_h1,
                                       self.nn.n_h2, self.n_agents, self.n_actions)
            observations = self.obs_space.sample()[np.newaxis]
            observations = jnp.array([observations for _ in 
                                                    range(self.n_agents)])
            observations = jnp.transpose(observations, axes=(1, 0, 2, 3, 4))

            action_1hot = MultiDiscrete([self.n_actions_for_r for i  
                                             in range(self.n_agents)]).sample()[np.newaxis]
            action_1hot = util.get_action_1hot_flat_batch(action_1hot,
                                                         self.n_actions_for_r)       
            critic = Model.create(critic_def,
                              inputs=[critic_key, observations],
                              tx=optax.adam(learning_rate=self.lr_v_model))
            target_critic = Model.create(critic_def,
                              inputs=[critic_key, observations],
                              tx=optax.adam(learning_rate=self.lr_v_model))

            rewarder = Model.create(rewarder_def,
                            inputs=[rewarder_key, observations,
                            action_1hot],
                            tx=optax.adam(learning_rate=self.lr_reward))
        else:
            critic_def = nets.CentralizedVNetMLP(self.nn.n_h1, self.nn.n_h2,
                                   self.n_agents, self.n_actions)
            rewarder_def = nets.RewardMLP(self.nn.n_h1, self.nn.n_h2, self.n_agents, self.n_actions_for_r)
            observations = self.obs_space.sample()[np.newaxis]
            observations = jnp.array([observations for _ in
                                        range(self.n_agents)])
            observations = jnp.transpose(observations, axes=(1, 0, 2))
            action_1hot = MultiDiscrete([self.n_actions_for_r for i  
                                             in range(self.n_agents)]).sample()[np.newaxis]
            action_1hot = util.get_action_1hot_flat_batch(action_1hot,
                                                         self.n_actions_for_r)     
            critic = Model.create(critic_def,
                              inputs=[critic_key, observations],
                              tx=optax.adam(learning_rate=self.lr_v_model))
            target_critic = Model.create(critic_def,
                              inputs=[critic_key, observations],
                              tx=optax.adam(learning_rate=self.lr_v_model))
            rewarder = Model.create(rewarder_def,
                            inputs=[rewarder_key, observations,
                            action_1hot],
                            tx=optax.adam(learning_rate=self.lr_reward))

        self.critic = critic
        self.target_critic = target_critic
        self.rewarder = rewarder
        self.rng = rng

    def receive_list_of_agents(self, list_of_agents):
        self.list_of_agents = list_of_agents

    def give_reward(self, obses, action_all):

        action_1hot_flat = util.get_action_1hot_flat(action_all, self.n_actions_for_r)

        if len(jnp.array(action_all).shape) == 1:
            action_1hot_flat = jnp.expand_dims(action_1hot_flat, axis=0)

        obses = jnp.array(obses)

        reward = single_incentivize(self.rewarder, obses, action_1hot_flat,
                             self.r_multiplier)
        return reward.squeeze()

    def train_reward(self, list_old_actor_params,
                     list_bufs, list_bufs_new, epsilon):

        old_batches = Batch(*[jnp.array(getattr(list_bufs, name)) \
                            for name in Batch._fields])

        new_critic, new_target_critic, critic_info = _update_jit(
            self.critic, self.target_critic, 
            old_batches, self.gamma, self.tau)

        self.critic = new_critic
        self.target_critic = new_target_critic

        new_batches = Batch(*[jnp.array(getattr(list_bufs_new, name)) \
                            for name in Batch._fields])
        actors = tuple([agent.actor for agent in self.list_of_agents])
        critics = tuple([agent.critic for agent in self.list_of_agents])
        lr_acs = tuple([agent.lr_actor for agent in self.list_of_agents])
        new_rewarder, mech_info = update_rewarder(self.rewarder.params,
                        tuple(list_old_actor_params),
                        actors[0],
                        self.critic, critics, self.rewarder,
                        epsilon, self.reg_coeff, old_batches, new_batches, 
                        lr_acs, 
                        self.n_actions_for_r, 
                        self.r_multiplier, self.entropy_coeff,
                        self.gamma)

        self.rewarder = new_rewarder
        return {**mech_info, **critic_info}


@jax.partial(jax.jit, static_argnums=(3,))
def incentivize(rewarder: Model, obs: jnp.ndarray, 
                action_all: jnp.ndarray,
                r_multiplier: float,
                rewarder_params: Optional[FrozenDict] = None
                ) -> jnp.ndarray:
    if rewarder_params is None:
        rewarder_params = rewarder.params
    incentives = rewarder.apply({'params': rewarder_params}, obs, action_all)
    if len(obs.shape) == 4:
        a_s = obs.shape[0]
        incentives = incentives.reshape(1, a_s) * r_multiplier
    else:
        b_s, a_s = obs.shape[0], obs.shape[1]
        incentives = incentives.reshape(b_s, a_s) * r_multiplier

    return incentives


@jax.partial(jax.jit, static_argnums=(3,))
def single_incentivize(rewarder: Model, obs: jnp.ndarray, 
                action_all: jnp.ndarray,
                r_multiplier: float,
                rewarder_params: Optional[FrozenDict] = None
                ) -> jnp.ndarray:
    if rewarder_params is None:
        rewarder_params = rewarder.params
    incentives = rewarder.apply({'params': rewarder_params}, obs, action_all)
    a_s = obs.shape[0]
    incentives = incentives.reshape(1, a_s) * r_multiplier

    return incentives


def update_critic(critic: Model, target_critic: Model, 
                  batch: Batch, discount: float
                  ) -> Tuple[Model, InfoDict]:
    obs_next = jnp.array(batch.obs_next)

    obs = jnp.array(batch.obs)
    
    v_target_next = target_critic(obs_next)

    total_reward = batch.reward.sum(axis=-1)
    done = jnp.all(batch.done, -1)

    td_target = total_reward + discount * (1 - done) * v_target_next.squeeze(1)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = critic.apply({'params': critic_params}, obs)
        critic_loss = ((v.squeeze(1) - td_target)**2).mean(0)
    
        return critic_loss, {
            'critic_loss': critic_loss,
            'v': v.mean(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


# @jax.partial(jax.jit, static_argnums=(11, 12, 13, 14, 15))
def loss_rewarder(main_rewarder_params: FrozenDict,
                    old_actor_params: Sequence[FrozenDict],
                    actor: Model,
                    critic: Model,
                    critics: Sequence[Model],
                    rewarder: Model,
                    epsilon: float,
                    reg_coeff: float,
                    old_batches: Batch,
                    new_batches: Batch,
                    lr_acs: Sequence[float],
                    n_actions_for_r: int,
                    r_multiplier: float,
                    entropy_coeff: float,
                    discount: float):
    action_1hot_flat = util.get_action_1hot_flat_batch(old_batches.action_all, 
                                                        n_actions_for_r)
    old_observations = jnp.array(old_batches.obs)

    incentives = incentivize(rewarder, old_observations,
                                action_1hot_flat,
                                r_multiplier, main_rewarder_params)
    
    def actor_loss_fn(actor_params: Params, obs: jnp.ndarray,
                      actions: jnp.ndarray, dones: jnp.ndarray,
                      v_next: jnp.ndarray, value: jnp.ndarray,
                      reward: jnp.ndarray) \
                      -> Tuple[jnp.ndarray, InfoDict]:
        log_probs = actor.apply({'params': actor_params}, obs, 
                                epsilon, return_logits=False)
        probs = jnp.exp(log_probs)

        actions_1hot = util.process_actions(actions, \
                            actor.apply_fn.n_actions)

        log_probs_taken = jnp.log(jnp.multiply(probs, actions_1hot).sum(axis=1) + 1e-15)
        
        entropy = -(probs * log_probs).sum(-1)
        td_error = reward.reshape(-1, 1) + discount * (1 - dones).reshape(-1, 1) * v_next - value
        policy_loss = -jnp.multiply(log_probs_taken, td_error.squeeze(-1))
        actor_loss = (policy_loss - entropy_coeff * entropy).sum(0)
        return actor_loss

    def simple_actor_loss_fn(actor_params: Params, obs: jnp.ndarray,
                      actions: jnp.ndarray, dones: jnp.ndarray,
                      v_next: jnp.ndarray, value: jnp.ndarray,
                      reward: jnp.ndarray) \
                      -> Tuple[jnp.ndarray, InfoDict]:
        log_probs = actor.apply({'params': actor_params}, obs, 
                                epsilon, return_logits=False)
        probs = jnp.exp(log_probs)

        actions_1hot = util.process_actions(actions, \
                            actor.apply_fn.n_actions)

        log_probs_taken = jnp.log(jnp.multiply(probs, actions_1hot).sum(axis=1) + 1e-15)
        
        td_error = reward.reshape(-1, 1) + discount * (1 - dones).reshape(-1, 1) * v_next - value
        policy_loss = -jnp.multiply(log_probs_taken, td_error.squeeze(-1))
        actor_loss = (policy_loss).sum(0)
        return actor_loss

    new_actor_params = []

    for idx, actor_param in enumerate(old_actor_params):
        value = critics[idx](old_batches.obs[:, idx, ...])
        v_next = critics[idx](old_batches.obs_next[:, idx, ...])
        total_reward = old_batches.reward[:, idx, ...]+ incentives[..., idx]
        actor_loss_grad_fn = jax.grad(actor_loss_fn, argnums=0)
        grads = actor_loss_grad_fn(actor_param, old_batches.obs[:, idx, ...],
                                   old_batches.action[:, idx, ...], 
                                   old_batches.done[:, idx, ...], v_next,
                                   value, total_reward)
        new_actor_param = jax.tree_multimap(
                        lambda p, g: p - lr_acs[idx] * g, actor_param,
                        grads)
        new_actor_params.append(new_actor_param)

    def meta_loss_fn(new_actor_params: Sequence[FrozenDict]):

        losses = []
        env_reward_meta = new_batches.reward.sum(axis=-1)
        new_obs = jnp.array(new_batches.obs)
        new_obs_next = jnp.array(new_batches.obs_next)
        value_mech = critic(new_obs)
        v_next_mech = critic(new_obs_next)
        dones = jnp.all(new_batches.done, -1)

        for idx, actor_param in enumerate(new_actor_params):
            meta_loss = simple_actor_loss_fn(actor_param, new_batches.obs[:, idx, ...],
                                    new_batches.action[:, idx, ...], 
                                    dones, v_next_mech,
                                    value_mech, env_reward_meta)

            losses.append(meta_loss)

        rewarder_loss = jnp.array(losses).sum()
        incentive_cost = jnp.abs(incentives).sum(axis=-1)
        incentive_discount = jnp.cumprod(jnp.ones_like(incentive_cost) * discount) / discount
        incentive_cost = jnp.multiply(incentive_cost, incentive_discount).sum()
        rewarder_loss += reg_coeff * incentive_cost

        return rewarder_loss, {'rewarder_loss': rewarder_loss}

    return meta_loss_fn(new_actor_params)


@jax.partial(jax.jit, static_argnums=(10, 11, 12, 13, 14))
def update_rewarder(main_rewarder_params: FrozenDict,
                    old_actor_params: Sequence[FrozenDict],
                    actor: Model,
                    critic: Model,
                    critics: Sequence[Model],
                    rewarder: Model,
                    epsilon: float,
                    reg_coeff: float,
                    old_batches: Batch,
                    new_batches: Batch,
                    lr_acs: Sequence[float],
                    n_actions_for_r: int,
                    r_multiplier: float,
                    entropy_coeff: float,
                    discount: float):

    grad_fn = jax.grad(loss_rewarder, has_aux=True, argnums=0)
    grads, info = grad_fn(main_rewarder_params,
                    old_actor_params,
                    actor,
                    critic,
                    critics,
                    rewarder,
                    epsilon,
                    reg_coeff,
                    old_batches,
                    new_batches,
                    lr_acs,
                    n_actions_for_r,
                    r_multiplier,
                    entropy_coeff,
                    discount)

    updates, new_opt_state = rewarder.tx.update(grads, \
                                                rewarder.opt_state,
                                                rewarder.params)

    new_params = optax.apply_updates(rewarder.params, updates)

    return rewarder.replace(step=rewarder.step + 1,
                            params=new_params,
                            opt_state=new_opt_state), info


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_multimap(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)