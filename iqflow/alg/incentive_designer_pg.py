# Policy Gradient based Incentive Designer (ID) algorithm

from random import seed
from typing import Sequence, Tuple, Optional, Any

from numpy.lib.shape_base import expand_dims

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

from iqflow.networks.common import InfoDict, Model, Model2Optim, PRNGKey, Params, mini_init

from iqflow.alg.policy_gradient import PolicyGradient as Alg


Batch = collections.namedtuple(
    'Batch',
    ['obs', 'action', 'reward', 'obs_next', 'done',
     'r_from_mech', 'action_all'])



def _update_model_jit(
    actor: Model, batch: Batch) -> Tuple[Model, InfoDict]:

    new_actor, actor_info = update_actor_model(actor, batch)

    return new_actor, actor_info


class IDPG(object):

    def __init__(self, obs_space, config, dim_obs, n_actions, nn_config, agent_name,
                 r_multiplier, n_agents, n_actions_for_r, rng):
        self.alg_name = 'incentive_designer_pg'
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
        self.lr_actor = config.lr_actor
        self.lr_cost = config.lr_cost
        self.lr_reward = config.lr_reward
        self.reg_coeff = config.reg_coeff
        self.separate_cost_optimizer = config.separate_cost_optimizer
        self.config = config

        self.rng = rng

        self.create_networks()
        
    def create_networks(self):
        """Instantiates the neural network part of computation graph."""

        rng, rewarder_key = jax.random.split(self.rng, 2)

        if self.image_obs:
            rewarder_def = nets.Reward(self.nn.n_filters * self.n_agents,
                                       self.nn.kernel,
                                       self.nn.stride, self.nn.n_hr1,
                                       self.nn.n_hr2, self.n_agents, self.n_actions, nn.sigmoid)
            observations = self.obs_space.sample()[np.newaxis]
            observations = jnp.array([observations for _ in 
                                                    range(self.n_agents)])
            observations = jnp.transpose(observations, axes=(1, 0, 2, 3, 4))

            action_1hot = MultiDiscrete([self.n_actions_for_r for i  
                                             in range(self.n_agents)]).sample()[np.newaxis]
            action_1hot = util.get_action_1hot_flat_batch(action_1hot,
                                                         self.n_actions_for_r)

            if self.separate_cost_optimizer:
                rewarder = Model2Optim.create(rewarder_def,
                            inputs=[rewarder_key, observations,
                            action_1hot],
                            tx1=optax.adam(learning_rate=self.lr_reward),
                            tx2=optax.adam(learning_rate=self.lr_cost)
                            )
            else:
                rewarder = Model.create(rewarder_def,
                                inputs=[rewarder_key, observations,
                                action_1hot],
                                tx=optax.adam(learning_rate=self.lr_reward))
        else:
            # rewarder_def = nets.RewardMLP(self.nn.n_hr1, self.nn.n_hr2, self.n_agents, self.n_actions, nn.sigmoid) 
            rewarder_def = nets.RewardMLP(self.nn.n_hr1, self.nn.n_hr2, self.n_agents, self.n_actions, nets.abs_tanh, last_kernel_init=mini_init())

            observations = self.obs_space.sample()[np.newaxis]
            observations = jnp.array([observations for _ in
                                        range(self.n_agents)])
            observations = jnp.transpose(observations, axes=(1, 0, 2))
            action_others = MultiDiscrete([self.n_actions_for_r for i  
                                             in range(self.n_agents)]).sample()[np.newaxis]
            action_others = util.get_action_1hot_flat_batch(action_others,
                                                         self.n_actions_for_r)

            if self.separate_cost_optimizer:
                rewarder = Model2Optim.create(rewarder_def,
                                inputs=[rewarder_key, observations,
                                action_others],
                                tx1=optax.adam(learning_rate=self.lr_reward),
                                tx2=optax.adam(learning_rate=self.lr_cost)
                                )
            else:
                rewarder = Model.create(rewarder_def,
                                inputs=[rewarder_key, observations,
                                action_others],
                                tx=optax.adam(learning_rate=self.lr_reward))

        self.rewarder = rewarder
        self.rng = rng

    def receive_list_of_agents(self, list_of_agents):
        self.list_of_agents = list_of_agents

    def model_agents(self):
        self.list_of_agents = []
        for agent_id in range(self.n_agents):
            rng, agent_key = jax.random.split(self.rng, 2)
            self.list_of_agents.append(
                Alg(self.obs_space, self.config, 
                    self.dim_obs, self.n_actions,
                    self.nn, 'agent_%d' % agent_id,
                    agent_id, rng=agent_key))
            self.rng = rng

    def train_agent_models(self, list_bufs):
        batches = tuple([Batch(*[jnp.array(getattr(list_bufs, name))[:, idx] \
                            for name in Batch._fields]) \
                            for idx in range(self.n_agents)])
        actors = tuple([agent.actor for agent in self.list_of_agents])
        new_actors, infos = update_agent_models(actors, batches)
        for agent_id in range(self.n_agents):
            self.list_of_agents[agent_id].actor = new_actors[agent_id]

        return infos


    def give_reward(self, obses, action_all):

        action_1hot_flat = util.get_action_1hot_flat(action_all, self.n_actions)


        if len(jnp.array(action_all).shape) == 1:
            action_1hot_flat = jnp.expand_dims(action_1hot_flat, axis=0)

        obses = jnp.array(obses)
        dec_obses = obses

        reward = single_incentivize(self.rewarder, dec_obses, 
                             action_1hot_flat,
                             self.r_multiplier)
        return reward.squeeze()


    def train_reward(self, list_old_actor_params,
                     list_bufs, list_bufs_new):

        old_batches = Batch(*[jnp.array(getattr(list_bufs, name)) \
                            for name in Batch._fields])

        new_batches = Batch(*[jnp.array(getattr(list_bufs_new, name)) \
                            for name in Batch._fields])
        actors = tuple([agent.actor for agent in self.list_of_agents])
        lr_acs = tuple([agent.lr_actor for agent in self.list_of_agents])
        new_rewarder, mech_info = update_rewarder(self.rewarder.params,
                        tuple(list_old_actor_params),
                        actors[0], self.rewarder,
                        self.reg_coeff, old_batches, new_batches, 
                        lr_acs, 
                        self.n_actions_for_r, 
                        self.r_multiplier, self.entropy_coeff,
                        self.gamma,
                        self.separate_cost_optimizer)

        self.rewarder = new_rewarder
        return mech_info


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


def update_actor_model(actor: Model,
                 batch: Batch,
                 ) -> Tuple[Model, InfoDict]:

    actions_1hot = util.process_actions(batch.action, actor.apply_fn.n_actions)
    
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        logits, log_probs = actor.apply({'params': actor_params}, batch.obs, 
                                0, return_logits=True)
        actor_loss = optax.softmax_cross_entropy(logits, actions_1hot).mean()
        return actor_loss, {'actor_loss': actor_loss}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info


@jax.partial(jax.jit)
def update_agent_models(actors: Sequence[Model],
                        batches: Sequence[Batch]):

    new_actors = []
    new_infos = []
    for idx in range(len(actors)):
        new_actor, new_info = _update_model_jit(actors[idx],
                          batches[idx]
                        )
        new_actors.append(new_actor)
        new_infos.append(new_info)

    return new_actors, new_infos


def loss_rewarder(main_rewarder_params: FrozenDict,
                    old_actor_params: Sequence[FrozenDict],
                    actor: Model,
                    rewarder: Model,
                    reg_coeff: float,
                    old_batches: Batch,
                    new_batches: Batch,
                    lr_acs: Sequence[float],
                    n_actions_for_r: int,
                    r_multiplier: float,
                    entropy_coeff: float,
                    discount: float,
                    separate_cost_optim: bool):
    action_1hot_flat = util.get_action_1hot_flat_batch(old_batches.action_all, 
                                                        n_actions_for_r)
    old_observations = jnp.array(old_batches.obs)

    incentives = incentivize(rewarder, old_observations,
                                action_1hot_flat,
                                r_multiplier, main_rewarder_params)
    
    def actor_loss_fn(actor_params: Params, obs: jnp.ndarray,
                      actions: jnp.ndarray, returns: jnp.ndarray) \
                      -> Tuple[jnp.ndarray, InfoDict]:
        log_probs = actor.apply({'params': actor_params}, obs, 
                                return_logits=False)
        probs = jnp.exp(log_probs)

        actions_1hot = util.process_actions(actions, \
                            actor.apply_fn.n_actions)

        log_probs_taken = jnp.log(jnp.multiply(probs, actions_1hot).sum(axis=1) + 1e-15)
        
        entropy = -(probs * log_probs).sum(-1)
        policy_loss = -jnp.multiply(log_probs_taken, returns.reshape(*log_probs_taken.shape))
        actor_loss = (policy_loss - entropy_coeff * entropy).sum(0)
        return actor_loss

    def simple_actor_loss_fn(actor_params: Params, obs: jnp.ndarray,
                      actions: jnp.ndarray, returns: jnp.ndarray) \
                      -> Tuple[jnp.ndarray, InfoDict]:
        log_probs = actor.apply({'params': actor_params}, obs, 
                                return_logits=False)
        probs = jnp.exp(log_probs)

        actions_1hot = util.process_actions(actions, \
                            actor.apply_fn.n_actions)

        log_probs_taken = jnp.log(jnp.multiply(probs, actions_1hot).sum(axis=1) + 1e-15)

        policy_loss = -jnp.multiply(log_probs_taken, returns.squeeze(-1))
        actor_loss = (policy_loss).sum(0)
        return actor_loss

    new_actor_params = []

    for idx, actor_param in enumerate(old_actor_params):
        total_reward = old_batches.reward[:, idx, ...]+ incentives[..., idx]
        actor_loss_grad_fn = jax.grad(actor_loss_fn, argnums=0)
        ag_return_discount = jnp.cumprod(jnp.ones_like(old_batches.reward[:, idx, ...]+ incentives[..., idx]) * discount) / discount
        ag_returns = jnp.multiply(old_batches.reward[:, idx, ...]+ incentives[..., idx], ag_return_discount)
        grads = actor_loss_grad_fn(actor_param, old_batches.obs[:, idx, ...],
                                   old_batches.action[:, idx, ...], 
                                   ag_returns)
        new_actor_param = jax.tree_multimap(
                        lambda p, g: p - lr_acs[idx] * g, actor_param,
                        grads)
        new_actor_params.append(new_actor_param)

    def meta_loss_fn(new_actor_params: Sequence[FrozenDict]):

        losses = []
        env_reward_meta = new_batches.reward.sum(axis=-1)
        new_obs = jnp.array(new_batches.obs)

        new_obs_next = jnp.array(new_batches.obs_next)

        dones = jnp.all(new_batches.done, -1)
        return_discount = jnp.cumprod(jnp.ones_like(new_batches.reward.sum(axis=-1, keepdims=True)) * discount) / discount
        returns = jnp.multiply(new_batches.reward.sum(axis=-1, keepdims=True), return_discount.reshape(-1, 1))

        for idx, actor_param in enumerate(new_actor_params):
            meta_loss = simple_actor_loss_fn(actor_param, new_batches.obs[:, idx, ...],
                                    new_batches.action[:, idx, ...], 
                                    returns)

            losses.append(meta_loss)


        inc_discount = discount

        rewarder_loss = jnp.array(losses).sum()
        incentive_cost = jnp.abs(incentives).sum(axis=-1)
        incentive_discount = jnp.cumprod(jnp.ones_like(incentive_cost) * inc_discount) / inc_discount
        incentive_cost = jnp.multiply(incentive_cost, incentive_discount).sum()
        if not separate_cost_optim:
            rewarder_loss += reg_coeff * incentive_cost

        return rewarder_loss, {'rewarder_loss': rewarder_loss}

    return meta_loss_fn(new_actor_params)


def loss_cost(main_rewarder_params: FrozenDict,
                    old_actor_params: Sequence[FrozenDict],
                    actor: Model,
                    rewarder: Model,
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
                      actions: jnp.ndarray, returns: jnp.ndarray) \
                      -> Tuple[jnp.ndarray, InfoDict]:
        log_probs = actor.apply({'params': actor_params}, obs, 
                                return_logits=False)
        probs = jnp.exp(log_probs)

        actions_1hot = util.process_actions(actions, \
                            actor.apply_fn.n_actions)

        log_probs_taken = jnp.log(jnp.multiply(probs, actions_1hot).sum(axis=1) + 1e-15)
        
        entropy = -(probs * log_probs).sum(-1)
        policy_loss = -jnp.multiply(log_probs_taken, returns.reshape(*log_probs_taken.shape))
        actor_loss = (policy_loss - entropy_coeff * entropy).sum(0)
        return actor_loss

    def simple_actor_loss_fn(actor_params: Params, obs: jnp.ndarray,
                      actions: jnp.ndarray, returns: jnp.ndarray) \
                      -> Tuple[jnp.ndarray, InfoDict]:
        log_probs = actor.apply({'params': actor_params}, obs, 
                                return_logits=False)
        probs = jnp.exp(log_probs)

        actions_1hot = util.process_actions(actions, \
                            actor.apply_fn.n_actions)

        log_probs_taken = jnp.log(jnp.multiply(probs, actions_1hot).sum(axis=1) + 1e-15)

        policy_loss = -jnp.multiply(log_probs_taken, returns.squeeze(-1))
        actor_loss = (policy_loss).sum(0)
        return actor_loss

    new_actor_params = []

    for idx, actor_param in enumerate(old_actor_params):
        total_reward = old_batches.reward[:, idx, ...]+ incentives[..., idx]
        actor_loss_grad_fn = jax.grad(actor_loss_fn, argnums=0)
        ag_return_discount = jnp.cumprod(jnp.ones_like(old_batches.reward[:, idx, ...]+ incentives[..., idx]) * discount) / discount
        ag_returns = jnp.multiply(old_batches.reward[:, idx, ...]+ incentives[..., idx], ag_return_discount)
        grads = actor_loss_grad_fn(actor_param, old_batches.obs[:, idx, ...],
                                   old_batches.action[:, idx, ...], 
                                   ag_returns)
        new_actor_param = jax.tree_multimap(
                        lambda p, g: p - lr_acs[idx] * g, actor_param,
                        grads)
        new_actor_params.append(new_actor_param)

    def meta_loss_fn(new_actor_params: Sequence[FrozenDict]):

        losses = []
        env_reward_meta = new_batches.reward.sum(axis=-1)
        new_obs = jnp.array(new_batches.obs)

        new_obs_next = jnp.array(new_batches.obs_next)

        dones = jnp.all(new_batches.done, -1)
        return_discount = jnp.cumprod(jnp.ones_like(new_batches.reward.sum(axis=-1, keepdims=True)) * discount) / discount
        returns = jnp.multiply(new_batches.reward.sum(axis=-1, keepdims=True), return_discount.reshape(-1, 1))

        for idx, actor_param in enumerate(new_actor_params):
            meta_loss = simple_actor_loss_fn(actor_param, new_batches.obs[:, idx, ...],
                                    new_batches.action[:, idx, ...], 
                                    returns)

            losses.append(meta_loss)

        rewarder_loss = jnp.array(losses).sum()
        incentive_cost = jnp.abs(incentives).sum(axis=-1)
        incentive_discount = jnp.cumprod(jnp.ones_like(incentive_cost) * discount) / discount
        incentive_cost = jnp.multiply(incentive_cost, incentive_discount).sum()

        return incentive_cost, {'incentive_cost': incentive_cost}

    return meta_loss_fn(new_actor_params)


@jax.partial(jax.jit, static_argnums=(7, 8, 9, 10, 11, 12))
def update_rewarder(main_rewarder_params: FrozenDict,
                    old_actor_params: Sequence[FrozenDict],
                    actor: Model,
                    rewarder: Any,
                    reg_coeff: float,
                    old_batches: Batch,
                    new_batches: Batch,
                    lr_acs: Sequence[float],
                    n_actions_for_r: int,
                    r_multiplier: float,
                    entropy_coeff: float,
                    discount: float,
                    separate_cost_optim: bool):

    grad_fn = jax.grad(loss_rewarder, has_aux=True, argnums=0)
    grads, info = grad_fn(main_rewarder_params,
                    old_actor_params,
                    actor,
                    rewarder,
                    reg_coeff,
                    old_batches,
                    new_batches,
                    lr_acs,
                    n_actions_for_r,
                    r_multiplier,
                    entropy_coeff,
                    discount,
                    separate_cost_optim)

    if separate_cost_optim:
        updates, new_opt_state = rewarder.tx1.update(grads, \
                                                    rewarder.opt_state1,
                                                    rewarder.params)
        cost_grad_fn = jax.grad(loss_cost, has_aux=True, argnums=0)
        cost_grads, cost_info = cost_grad_fn(main_rewarder_params,
                        old_actor_params,
                        actor,
                        rewarder,
                        reg_coeff,
                        old_batches,
                        new_batches,
                        lr_acs,
                        n_actions_for_r,
                        r_multiplier,
                        entropy_coeff,
                        discount)
        cost_updates, new_cost_opt_state = rewarder.tx2.update(cost_grads, \
                                                rewarder.opt_state2,
                                                rewarder.params)

        total_updates = jax.tree_multimap(
            lambda up, cup: up + cup, updates,
            cost_updates)

        updates = total_updates

        new_params = optax.apply_updates(rewarder.params, updates)

        return rewarder.replace(step=rewarder.step + 1,
                                params=new_params,
                                opt_state1=new_opt_state,
                                opt_state2=new_cost_opt_state), info
    else:
        updates, new_opt_state = rewarder.tx.update(grads, \
                                                rewarder.opt_state,
                                                rewarder.params)

        new_params = optax.apply_updates(rewarder.params, updates)

        return rewarder.replace(step=rewarder.step + 1,
                                params=new_params,
                                opt_state=new_opt_state), info
