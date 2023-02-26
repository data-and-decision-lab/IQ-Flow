# IQ-Flow Algorithm

from typing import Sequence, Tuple, Optional
from copy import deepcopy
from numpy.core.numeric import indices
from numpy.lib.npyio import recfromtxt

from numpy.lib.shape_base import expand_dims

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
import flax.linen as nn
import collections
import os

from gym.spaces import MultiDiscrete

from iqflow.networks import nets

from iqflow.utils import utils as util

from iqflow.networks.common import InfoDict, Model, PRNGKey, Params, mini_init

from iqflow.utils.utils import Batch, MechReplay, asymmetric_loss


@jax.partial(jax.jit, static_argnums=(10, 11, 12))
def _update_jit(
        coop_qcritic: Model, target_coop_qcritic: Model, coop_vcritic: Model,
        env_model_qcritic: Model, target_env_model_qcritic: Model, env_model_vcritic: Model,
        rewarder_model_qcritic: Model, target_rewarder_model_qcritic: Model, rewarder_model_vcritic: Model,
        batch: Batch, discount: float, tau: float, 
        n_actions: int, incentives: jnp.array, expectile: float) -> \
        Tuple[Model, Model, Model, Model,
              Model, Model, Model, Model, 
              Model, InfoDict]:

    new_coop_qcritic, new_coop_vcritic, new_env_model_qcritic, new_env_model_vcritic, \
        new_rewarder_model_qcritic, new_rewarder_model_vcritic, \
        critic_info = update_critics(coop_qcritic, target_coop_qcritic, coop_vcritic,
                                    env_model_qcritic, target_env_model_qcritic, env_model_vcritic,
                                    rewarder_model_qcritic, target_rewarder_model_qcritic, rewarder_model_vcritic,
                                    batch, discount, n_actions, incentives, expectile)

    new_target_coop_qcritic = target_update(new_coop_qcritic, target_coop_qcritic, tau)
    new_target_env_model_qcritic = target_update(new_env_model_qcritic, target_env_model_qcritic, tau)
    new_target_rewarder_model_qcritic = target_update(new_rewarder_model_qcritic, target_rewarder_model_qcritic, tau)

    return new_coop_qcritic, new_target_coop_qcritic, new_coop_vcritic, \
            new_env_model_qcritic, new_target_env_model_qcritic, new_env_model_vcritic, \
            new_rewarder_model_qcritic, new_target_rewarder_model_qcritic, \
            new_rewarder_model_vcritic, \
            {**critic_info}


class IQFlow(object):

    def __init__(self, obs_space, config, dim_obs, n_actions, nn_config, agent_name,
                 r_multiplier, n_agents, rng):
        self.alg_name = 'iqflow'
        self.dim_obs = dim_obs
        self.image_obs = isinstance(self.dim_obs, list)
        self.n_actions = n_actions
        self.nn = nn_config
        self.agent_name = agent_name
        self.r_multiplier = r_multiplier
        self.n_agents = n_agents
        self.obs_space = obs_space

        self.entropy_coeff = config.entropy_coeff
        self.gamma = config.gamma
        self.lr_reward = config.lr_reward
        self.lr_v_model = config.lr_v_model
        self.lr_v_rewarder = config.lr_v_rewarder
        self.cost_reg_coeff = config.cost_reg_coeff
        self.cost_reg_coeff2 = config.cost_reg_coeff2
        self.tau = config.tau
        self.config = config
        self.embed_dim = config.embed_dim
        self.expectile = config.expectile
        self.use_mini_init = config.use_mini_init
        self.cost_type = config.cost_type
        self.obj_mask = config.obj_mask

        self.replay_buffer = MechReplay(obs_space, n_actions, n_agents,
                                        config.batch_size, config.mech_buffer_size) 

        self.rng = rng

        self.create_networks()

    def create_networks(self):

        rng, coop_qcritic_key, coop_vcritic_key, env_model_qcritic_key, \
            env_model_vcritic_key, rewarder_model_qcritic_key, \
                rewarder_model_vcritic_key, \
                    rewarder_key = jax.random.split(self.rng, 8)

        if self.image_obs:
            coop_qcritic_def = nets.QNetNAgent(self.nn.n_filters,
                                   self.nn.kernel, self.nn.stride,
                                   self.nn.n_h1, self.nn.n_h2,
                                   self.n_agents, self.n_actions)

            coop_vcritic_def = nets.CentralizedVNet(self.nn.n_filters,
                                   self.nn.kernel, self.nn.stride,
                                   self.nn.n_h1, self.nn.n_h2,
                                   self.n_agents, self.n_actions)

            env_model_qcritic_def = nets.ModelQNetNAgent(self.nn.n_filters,
                                         self.nn.kernel, self.nn.stride,
                                         self.nn.n_h1, self.nn.n_h2,
                                         self.n_agents, self.n_actions)

            env_model_vcritic_def = nets.CentralizedInterestVNet(self.nn.n_filters,
                                         self.nn.kernel, self.nn.stride,
                                         self.nn.n_h1, self.nn.n_h2,
                                         self.n_agents, self.n_actions)

            rewarder_model_qcritic_def = nets.ModelQNetNAgent(self.nn.n_filters,
                                         self.nn.kernel, self.nn.stride,
                                         self.nn.n_h1, self.nn.n_h2,
                                         self.n_agents, self.n_actions)

            rewarder_model_vcritic_def = nets.CentralizedInterestVNet(self.nn.n_filters,
                                         self.nn.kernel, self.nn.stride,
                                         self.nn.n_h1, self.nn.n_h2,
                                         self.n_agents, self.n_actions)

            if self.use_mini_init:    
                rewarder_def = nets.Reward(self.nn.n_filters,
                                        self.nn.kernel, self.nn.stride, 
                                        self.nn.n_h1, self.nn.n_h2, 
                                        self.n_agents, self.n_actions,
                                        last_kernel_init=mini_init())
            else:
                rewarder_def = nets.Reward(self.nn.n_filters,
                                        self.nn.kernel, self.nn.stride, 
                                        self.nn.n_h1, self.nn.n_h2, 
                                        self.n_agents, self.n_actions,
                                        output_nonlinearity=nn.sigmoid)

            observations = self.obs_space.sample()[np.newaxis]
            observations = jnp.array([observations for _ in
                                        range(self.n_agents)])
            dec_observations = jnp.transpose(observations, axes=(1, 0, 2, 3, 4))

        else:
            coop_qcritic_def = nets.QNetMLPNAgent(self.nn.n_h1, self.nn.n_h2,
                                   self.n_agents, self.n_actions)

            coop_vcritic_def = nets.CentralizedVNetMLP(self.nn.n_h1, self.nn.n_h2,
                                   self.n_agents, self.n_actions)

            env_model_qcritic_def = nets.ModelQNetMLPNAgent(self.nn.n_h1, self.nn.n_h2,
                                   self.n_agents, self.n_actions)

            env_model_vcritic_def = nets.CentralizedInterestVNetMLP(self.nn.n_h1, self.nn.n_h2,
                                         self.n_agents, self.n_actions)

            rewarder_model_qcritic_def = nets.ModelQNetMLPNAgent(self.nn.n_h1, self.nn.n_h2,
                                   self.n_agents, self.n_actions)

            rewarder_model_vcritic_def = nets.CentralizedInterestVNetMLP(self.nn.n_h1, self.nn.n_h2,
                                         self.n_agents, self.n_actions)


            if self.use_mini_init:  
                rewarder_def = nets.RewardMLP(self.nn.n_hr1, self.nn.n_hr2, 
                                       self.n_agents, self.n_actions, last_kernel_init=mini_init())
            else:
                rewarder_def = nets.RewardMLP(self.nn.n_hr1, self.nn.n_hr2, 
                                       self.n_agents, self.n_actions, output_nonlinearity=nn.sigmoid)

            observations = self.obs_space.sample()[np.newaxis]
            observations = jnp.array([observations for _ in
                                        range(self.n_agents)])
            dec_observations = jnp.transpose(observations, axes=(1, 0, 2))

        action_1hot = MultiDiscrete([self.n_actions for i
                                        in range(self.n_agents)]).sample()[np.newaxis]
        action_1hot = util.get_action_1hot_flat_batch(action_1hot,
                                                    self.n_actions)
        coop_qcritic = Model.create(coop_qcritic_def,
                                inputs=[coop_qcritic_key, dec_observations, action_1hot],
                                tx=optax.adam(learning_rate=self.lr_v_model))

        target_coop_qcritic = Model.create(coop_qcritic_def,
                                        inputs=[coop_qcritic_key, dec_observations, action_1hot],
                                        tx=optax.adam(learning_rate=self.lr_v_model))
        coop_vcritic = Model.create(coop_vcritic_def,
                                inputs=[coop_vcritic_key, dec_observations],
                                tx=optax.adam(learning_rate=self.lr_v_model))
        
        env_model_qcritic = Model.create(env_model_qcritic_def,
                                    inputs=[env_model_qcritic_key, dec_observations, action_1hot],
                                    tx=optax.rmsprop(learning_rate=self.lr_v_model))
        target_env_model_qcritic = Model.create(env_model_qcritic_def,
                                            inputs=[env_model_qcritic_key, dec_observations, action_1hot],
                                            tx=optax.rmsprop(learning_rate=self.lr_v_model))
        env_model_vcritic = Model.create(env_model_vcritic_def,
                                    inputs=[env_model_vcritic_key, dec_observations],
                                    tx=optax.rmsprop(learning_rate=self.lr_v_model))

        rewarder_model_qcritic = Model.create(rewarder_model_qcritic_def,
                                    inputs=[rewarder_model_qcritic_key, dec_observations, action_1hot],
                                    tx=optax.rmsprop(learning_rate=self.lr_v_rewarder))
        target_rewarder_model_qcritic = Model.create(rewarder_model_qcritic_def,
                                            inputs=[rewarder_model_qcritic_key, dec_observations, action_1hot],
                                            tx=optax.rmsprop(learning_rate=self.lr_v_rewarder))
        rewarder_model_vcritic = Model.create(rewarder_model_vcritic_def,
                                    inputs=[rewarder_model_vcritic_key, dec_observations],
                                    tx=optax.rmsprop(learning_rate=self.lr_v_rewarder))

        rewarder = Model.create(rewarder_def,
                                inputs=[rewarder_key, dec_observations,
                                        action_1hot],
                                tx=optax.rmsprop(learning_rate=self.lr_reward))

        self.coop_qcritic = coop_qcritic
        self.target_coop_qcritic = target_coop_qcritic
        self.coop_vcritic = coop_vcritic
        self.env_model_qcritic = env_model_qcritic
        self.target_env_model_qcritic = target_env_model_qcritic
        self.env_model_vcritic = env_model_vcritic
        self.rewarder_model_qcritic = rewarder_model_qcritic
        self.target_rewarder_model_qcritic = target_rewarder_model_qcritic
        self.rewarder_model_vcritic = rewarder_model_vcritic
        self.rewarder = rewarder
        self.rng = rng

    def reset_rewarder_critic(self):
        rng, rewarder_model_qcritic_key, \
            rewarder_model_vcritic_key = jax.random.split(self.rng, 3)

        if self.image_obs:
            rewarder_model_qcritic_def = nets.ModelQNetNAgent(self.nn.n_filters,
                                         self.nn.kernel, self.nn.stride,
                                         self.nn.n_h1, self.nn.n_h2,
                                         self.n_agents, self.n_actions)

            rewarder_model_vcritic_def = nets.CentralizedInterestVNet(self.nn.n_filters,
                                         self.nn.kernel, self.nn.stride,
                                         self.nn.n_h1, self.nn.n_h2,
                                         self.n_agents, self.n_actions)

            observations = self.obs_space.sample()[np.newaxis]
            observations = jnp.array([observations for _ in
                                        range(self.n_agents)])
            dec_observations = jnp.transpose(observations, axes=(1, 0, 2, 3, 4))

        else:
            rewarder_model_qcritic_def = nets.ModelQNetMLPNAgent(self.nn.n_h1, self.nn.n_h2,
                                   self.n_agents, self.n_actions)

            rewarder_model_vcritic_def = nets.CentralizedInterestVNetMLP(self.nn.n_h1, self.nn.n_h2,
                                         self.n_agents, self.n_actions)

            observations = self.obs_space.sample()[np.newaxis]
            observations = jnp.array([observations for _ in
                                        range(self.n_agents)])
            dec_observations = jnp.transpose(observations, axes=(1, 0, 2))

        action_1hot = MultiDiscrete([self.n_actions for i
                                        in range(self.n_agents)]).sample()[np.newaxis]
        action_1hot = util.get_action_1hot_flat_batch(action_1hot,
                                                    self.n_actions)

        rewarder_model_qcritic = Model.create(rewarder_model_qcritic_def,
                                    inputs=[rewarder_model_qcritic_key, dec_observations, action_1hot],
                                    tx=optax.rmsprop(learning_rate=self.lr_v_rewarder))
        target_rewarder_model_qcritic = Model.create(rewarder_model_qcritic_def,
                                            inputs=[rewarder_model_qcritic_key, dec_observations, action_1hot],
                                            tx=optax.rmsprop(learning_rate=self.lr_v_rewarder))
        rewarder_model_vcritic = Model.create(rewarder_model_vcritic_def,
                                    inputs=[rewarder_model_vcritic_key, dec_observations],
                                    tx=optax.rmsprop(learning_rate=self.lr_v_rewarder))

        self.rewarder_model_qcritic = rewarder_model_qcritic
        self.target_rewarder_model_qcritic = target_rewarder_model_qcritic
        self.rewarder_model_vcritic = rewarder_model_vcritic
        self.rng = rng

    def insert_trajectory(self, buffer):
        buffer = deepcopy(buffer)
        action_1hot_flat = util.get_action_1hot_flat_batch(jnp.array(buffer.action_all), self.n_actions)
        action_1hot = util.get_action_1hot_batch(jnp.array(buffer.action_all), self.n_actions)
        self.replay_buffer.insert(buffer.obs, buffer.action, buffer.reward,
                            buffer.obs_next, buffer.done, buffer.action_all,
                            action_1hot_flat,
                            action_1hot, buffer.r_from_mech)
        return

    def give_reward(self, obses, action_all):

        action_1hot_flat = util.get_action_1hot_flat(action_all, self.n_actions)

        obses = jnp.array(obses)

        reward = single_incentivize(self.rewarder, obses, 
                             action_1hot_flat,
                             self.r_multiplier)
        return reward.squeeze()


    def train_critic(self):
        if self.replay_buffer.size < self.config.batch_size:
            return None
        batches = self.replay_buffer.sample(self.config.batch_size)

        obs = jnp.array(batches.obs)
  
        incentives = incentivize(self.rewarder, obs, batches.action_1hot_flat,
                                 self.r_multiplier)

        new_coop_qcritic, new_target_coop_qcritic, new_coop_vcritic, \
            new_env_model_qcritic, new_target_env_model_qcritic, new_env_model_vcritic, \
            new_rewarder_model_qcritic, new_target_rewarder_model_qcritic, \
            new_rewarder_model_vcritic, \
            critic_info = _update_jit(
                        self.coop_qcritic, self.target_coop_qcritic, self.coop_vcritic,
                        self.env_model_qcritic, self.target_env_model_qcritic, self.env_model_vcritic,
                        self.rewarder_model_qcritic, self.target_rewarder_model_qcritic, 
                        self.rewarder_model_vcritic, batches, self.gamma, self.tau, self.n_actions,
                        incentives, self.expectile
                )

        self.coop_qcritic = new_coop_qcritic
        self.target_coop_qcritic = new_target_coop_qcritic
        self.coop_vcritic = new_coop_vcritic
        self.env_model_qcritic = new_env_model_qcritic
        self.target_env_model_qcritic = new_target_env_model_qcritic
        self.env_model_vcritic = new_env_model_vcritic
        self.rewarder_model_qcritic = new_rewarder_model_qcritic
        self.target_rewarder_model_qcritic = new_target_rewarder_model_qcritic
        self.rewarder_model_vcritic = new_rewarder_model_vcritic

        critic_info['rewarder_model_vcritic_loss'].block_until_ready()

        return critic_info

    def correct_model_critic(self):
        self.reset_rewarder_critic()
        batches = self.replay_buffer.sample(self.config.batch_size)

        obs = jnp.array(batches.obs)

        incentives = incentivize(self.rewarder, obs, batches.action_1hot_flat,
                                self.r_multiplier)

        new_coop_qcritic, new_target_coop_qcritic, new_coop_vcritic, \
            new_env_model_qcritic, new_target_env_model_qcritic, new_env_model_vcritic, \
            new_rewarder_model_qcritic, new_target_rewarder_model_qcritic, \
            new_rewarder_model_vcritic, \
            critic_info = _update_jit(
                        self.coop_qcritic, self.target_coop_qcritic, self.coop_vcritic,
                        self.env_model_qcritic, self.target_env_model_qcritic, self.env_model_vcritic,
                        self.rewarder_model_qcritic, self.target_rewarder_model_qcritic, 
                        self.rewarder_model_vcritic, batches, self.gamma, self.tau, self.n_actions,
                        incentives, self.expectile
                )
        for i in range(149):
            batches = self.replay_buffer.sample(self.config.batch_size)

            obs = jnp.array(batches.obs)
    
            incentives = incentivize(self.rewarder, obs, batches.action_1hot_flat,
                                    self.r_multiplier)

            new_coop_qcritic, new_target_coop_qcritic, new_coop_vcritic, \
                new_env_model_qcritic, new_target_env_model_qcritic, new_env_model_vcritic, \
                new_rewarder_model_qcritic, new_target_rewarder_model_qcritic, \
                new_rewarder_model_vcritic, \
                critic_info = _update_jit(
                            self.coop_qcritic, self.target_coop_qcritic, self.coop_vcritic,
                            self.env_model_qcritic, self.target_env_model_qcritic, self.env_model_vcritic,
                            new_rewarder_model_qcritic, new_target_rewarder_model_qcritic, 
                            new_rewarder_model_vcritic, batches, self.gamma, self.tau, self.n_actions,
                            incentives, self.expectile
                    )

            self.rewarder_model_qcritic = new_rewarder_model_qcritic
            self.target_rewarder_model_qcritic = new_target_rewarder_model_qcritic
            self.rewarder_model_vcritic = new_rewarder_model_vcritic

    def train(self, deneme=False):
        batches = self.replay_buffer.sample()

        obs = jnp.array(batches.obs)
  
        incentives = incentivize(self.rewarder, obs, batches.action_1hot_flat,
                                 self.r_multiplier)

        new_coop_qcritic, new_target_coop_qcritic, new_coop_vcritic, \
            new_env_model_qcritic, new_target_env_model_qcritic, new_env_model_vcritic, \
            new_rewarder_model_qcritic, new_target_rewarder_model_qcritic, \
            new_rewarder_model_vcritic, \
            critic_info = _update_jit(
                        self.coop_qcritic, self.target_coop_qcritic, self.coop_vcritic,
                        self.env_model_qcritic, self.target_env_model_qcritic, self.env_model_vcritic,
                        self.rewarder_model_qcritic, self.target_rewarder_model_qcritic, 
                        self.rewarder_model_vcritic, batches, self.gamma, self.tau, self.n_actions,
                        incentives, self.expectile
                )
        critic_info['rewarder_model_qcritic_loss'].block_until_ready()

        old_batches, new_batches = self.replay_buffer.meta_sample(self.config.meta_batch_size)


        new_rewarder, mech_info = update_rewarder(self.rewarder.params, self.rewarder, 
                                                    old_batches, new_batches, self.n_actions, 
                                                    self.r_multiplier, self.gamma,
                                                    new_coop_qcritic, new_env_model_qcritic,
                                                    new_env_model_vcritic,
                                                    new_rewarder_model_qcritic, 
                                                    new_rewarder_model_vcritic,
                                                    self.cost_reg_coeff, self.cost_reg_coeff2, 
                                                    self.cost_type, self.obj_mask)

        self.coop_qcritic = new_coop_qcritic
        self.target_coop_qcritic = new_target_coop_qcritic
        self.coop_vcritic = new_coop_vcritic
        self.env_model_qcritic = new_env_model_qcritic
        self.target_env_model_qcritic = new_target_env_model_qcritic
        self.env_model_vcritic = new_env_model_vcritic
        self.rewarder_model_qcritic = new_rewarder_model_qcritic
        self.target_rewarder_model_qcritic = new_target_rewarder_model_qcritic
        self.rewarder_model_vcritic = new_rewarder_model_vcritic

        self.rewarder = new_rewarder

        mech_info['rewarder_loss'].block_until_ready()

        return {**critic_info, **mech_info}


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


def update_critics(coop_qcritic: Model, target_coop_qcritic: Model, coop_vcritic: Model,
                   env_model_qcritic: Model, target_env_model_qcritic: Model, env_model_vcritic: Model,
                   rewarder_model_qcritic: Model, target_rewarder_model_qcritic: Model, rewarder_model_vcritic: Model,
                   batch: Batch, discount: float, n_actions: int, incentives: jnp.array, expectile: float
                   ) -> Tuple[Model, Model, Model, Model, Model, Model, InfoDict]:    

    dec_obs_next = jnp.array(batch.obs_next)
    dec_obs = jnp.array(batch.obs)
    actions = jnp.expand_dims(batch.action, axis=-1)

    coop_vtarget_next = coop_vcritic(dec_obs_next)
    coop_qcritic_td_target = batch.reward.sum(axis=-1, keepdims=True) + \
                                discount * (1 - jnp.all(batch.done == True, 
                                axis=-1, keepdims=True)) * coop_vtarget_next

    env_model_vtarget_next = env_model_vcritic(dec_obs_next)
    env_model_qcritic_td_target = batch.reward + discount * (1 - batch.done) * env_model_vtarget_next

    rewarder_model_vtarget_next = rewarder_model_vcritic(dec_obs_next)
    rewarder_model_qcritic_td_target = batch.reward + incentives + discount * (1 - batch.done) * rewarder_model_vtarget_next


    def coop_qcritic_loss_fn(coop_qcritic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = coop_qcritic.apply({'params': coop_qcritic_params}, dec_obs, batch.action_1hot_flat)
        v = v.reshape(v.shape[:-1] + (v.shape[-1] // n_actions, n_actions))
        v = jnp.take_along_axis(v, indices=actions, axis=-1).squeeze(-1)

        coop_qcritic_loss = ((v - jax.lax.stop_gradient(coop_qcritic_td_target))**2).mean()

        return coop_qcritic_loss, {
            'coop_qcritic_loss': coop_qcritic_loss,
            'coop_qcritic_v': v.mean(),
        }


    def coop_vcritic_loss_fn(coop_vcritic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        target_q = target_coop_qcritic(dec_obs, batch.action_1hot_flat)
        target_q = target_q.reshape(target_q.shape[:-1] + (target_q.shape[-1] // n_actions, n_actions))
        target_q= jnp.take_along_axis(target_q, indices=actions, axis=-1).squeeze(-1)
        v = coop_vcritic.apply({'params': coop_vcritic_params}, dec_obs)

        coop_vcritic_loss = asymmetric_loss(jax.lax.stop_gradient(target_q) - v, expectile).mean()

        return coop_vcritic_loss, {
            'coop_vcritic_loss': coop_vcritic_loss,
            'coop_vcritic_v': v.mean(),
        }


    def env_model_qcritic_loss_fn(env_model_qcritic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = env_model_qcritic.apply({'params': env_model_qcritic_params}, dec_obs, batch.action_1hot_flat)
        v = v.reshape(v.shape[:-1] + (v.shape[-1] // n_actions, n_actions))
        v = jnp.take_along_axis(v, indices=actions, axis=-1).squeeze(-1)

        env_model_qcritic_loss = ((v - jax.lax.stop_gradient(env_model_qcritic_td_target))**2).mean()

        return env_model_qcritic_loss, {
            'env_model_qcritic_loss': env_model_qcritic_loss,
            'env_model_qcritic_v': v.mean(),
        }


    def env_model_vcritic_loss_fn(env_model_vcritic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        target_q = target_env_model_qcritic(dec_obs, batch.action_1hot_flat)
        target_q = target_q.reshape(target_q.shape[:-1] + (target_q.shape[-1] // n_actions, n_actions))
        target_q= jnp.take_along_axis(target_q, indices=actions, axis=-1).squeeze(-1)
        v = env_model_vcritic.apply({'params': env_model_vcritic_params}, dec_obs)

        env_model_vcritic_loss = asymmetric_loss(jax.lax.stop_gradient(target_q) - v, expectile).mean()

        return env_model_vcritic_loss, {
            'env_model_vcritic_loss': env_model_vcritic_loss,
            'env_model_vcritic_v': v.mean(),
        }


    def rewarder_model_qcritic_loss_fn(rewarder_model_qcritic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = rewarder_model_qcritic.apply({'params': rewarder_model_qcritic_params}, dec_obs, batch.action_1hot_flat)
        v = v.reshape(v.shape[:-1] + (v.shape[-1] // n_actions, n_actions))
        v = jnp.take_along_axis(v, indices=actions, axis=-1).squeeze(-1)

        rewarder_model_qcritic_loss = ((v - jax.lax.stop_gradient(rewarder_model_qcritic_td_target))**2).mean()

        return rewarder_model_qcritic_loss, {
            'rewarder_model_qcritic_loss': rewarder_model_qcritic_loss,
            'rewarder_model_qcritic_v': v.mean(),
        }


    def rewarder_model_vcritic_loss_fn(rewarder_model_vcritic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        target_q = target_rewarder_model_qcritic(dec_obs, batch.action_1hot_flat)
        target_q = target_q.reshape(target_q.shape[:-1] + (target_q.shape[-1] // n_actions, n_actions))
        target_q= jnp.take_along_axis(target_q, indices=actions, axis=-1).squeeze(-1)
        v = rewarder_model_vcritic.apply({'params': rewarder_model_vcritic_params}, dec_obs)

        rewarder_model_vcritic_loss = asymmetric_loss(jax.lax.stop_gradient(target_q) - v, expectile).mean()

        return rewarder_model_vcritic_loss, {
            'rewarder_model_vcritic_loss': rewarder_model_vcritic_loss,
            'rewarder_model_vcritic_v': v.mean(),
        }


    new_coop_qcritic, new_coop_qcritic_info = coop_qcritic.apply_gradient(coop_qcritic_loss_fn)
    new_coop_vcritic, new_coop_vcritic_info = coop_vcritic.apply_gradient(coop_vcritic_loss_fn)
    new_env_model_qcritic, new_env_model_qcritic_info = env_model_qcritic.apply_gradient(env_model_qcritic_loss_fn)
    new_env_model_vcritic, new_env_model_vcritic_info = env_model_vcritic.apply_gradient(env_model_vcritic_loss_fn)
    new_rewarder_model_qcritic, new_rewarder_model_qcritic_info = rewarder_model_qcritic.apply_gradient(rewarder_model_qcritic_loss_fn)
    new_rewarder_model_vcritic, new_rewarder_model_vcritic_info = rewarder_model_vcritic.apply_gradient(rewarder_model_vcritic_loss_fn)
    

    return new_coop_qcritic, new_coop_vcritic, new_env_model_qcritic, new_env_model_vcritic, \
            new_rewarder_model_qcritic, new_rewarder_model_vcritic, \
            {**new_coop_qcritic_info, **new_coop_vcritic_info, 
             **new_env_model_qcritic_info, **new_env_model_vcritic_info, 
             **new_rewarder_model_qcritic_info,
             **new_rewarder_model_vcritic_info}


def  update_model_critics(batch: Batch, 
                         discount: float, 
                         n_actions: int,
                         incentives: jnp.array,
                         rewarder_model_qcritic: Model,
                         rewarder_model_vcritic: Model
                        ) -> Tuple[Model, InfoDict]:    

    dec_obs_next = jnp.array(batch.obs_next)
    dec_obs = jnp.array(batch.obs)
    actions = jnp.expand_dims(batch.action, axis=-1)

    rewarder_model_vtarget_next = jax.lax.stop_gradient(rewarder_model_vcritic(dec_obs_next))
    rewarder_model_qcritic_td_target = batch.reward + incentives + discount * (1 - batch.done) * rewarder_model_vtarget_next
    
    def rewarder_model_qcritic_loss_fn(rewarder_model_qcritic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = rewarder_model_qcritic.apply({'params': rewarder_model_qcritic_params}, dec_obs, batch.action_1hot_flat)
        v = v.reshape(v.shape[:-1] + (v.shape[-1] // n_actions, n_actions))
        v = jnp.take_along_axis(v, indices=actions, axis=-1).squeeze(-1)

        rewarder_model_qcritic_loss = ((v - rewarder_model_qcritic_td_target)**2).mean()

        return rewarder_model_qcritic_loss, {
            'rewarder_model_qcritic_loss': rewarder_model_qcritic_loss,
            'rewarder_model_qcritic_v': v.mean(),
        }

    new_rewarder_model_qcritic_params, new_rewarder_model_qcritic_info = rewarder_model_qcritic.apply_gradient_params(rewarder_model_qcritic_loss_fn)

    return new_rewarder_model_qcritic_params, \
           {**new_rewarder_model_qcritic_info}


def loss_rewarder(main_rewarder_params: FrozenDict,
                    rewarder: Model,
                    old_batches: Batch,
                    new_batches: Batch,
                    n_actions: int,
                    r_multiplier: float,
                    discount: float,
                    coop_qcritic: Model,
                    env_model_qcritic: Model,
                    env_model_vcritic: Model,
                    rewarder_model_qcritic: Model,
                    rewarder_model_vcritic: Model,
                    cost_reg_coeff: float,
                    cost_reg_coeff2: float,
                    cost_type: int,
                    obj_mask: bool):

    dec_old_observations = jnp.array(old_batches.obs)

    incentives = incentivize(rewarder, dec_old_observations, old_batches.action_1hot_flat,
                             r_multiplier, main_rewarder_params)

    for i in range(19):
        new_rewarder_model_qcritic_params, \
        critic_info = update_model_critics(old_batches,
                                            discount,
                                            n_actions,
                                            incentives,
                                            rewarder_model_qcritic,
                                            rewarder_model_vcritic)

        rewarder_model_qcritic = rewarder_model_qcritic.replace(params=new_rewarder_model_qcritic_params)


    new_rewarder_model_qcritic_params, \
    critic_info = update_model_critics(old_batches,
                                        discount,
                                        n_actions,
                                        incentives,
                                        rewarder_model_qcritic,
                                        rewarder_model_vcritic)
    

    def meta_loss_fn():
        dec_new_obs = new_batches.obs

        mech_v = coop_qcritic(dec_new_obs, new_batches.action_1hot_flat)
        mech_v = mech_v.reshape(mech_v.shape[:-1] + (mech_v.shape[-1] // n_actions, n_actions))

        model_v = env_model_qcritic.apply(
            {'params': env_model_qcritic.params}, dec_new_obs, new_batches.action_1hot_flat)
        model_v = model_v.reshape(model_v.shape[:-1] + (model_v.shape[-1] // n_actions, n_actions))

        rewarder_v = rewarder_model_qcritic.apply(
            {'params': new_rewarder_model_qcritic_params}, dec_new_obs, new_batches.action_1hot_flat)
        rewarder_v = rewarder_v.reshape(rewarder_v.shape[:-1] + (rewarder_v.shape[-1] // n_actions, n_actions))


        targets = mech_v.argmax(axis=-1)
        model_outputs = (rewarder_v).argmax(axis=-1)
        env_outputs = model_v.argmax(axis=-1)

        mask = (targets != model_outputs).astype(int).astype(float)
        mask = jnp.expand_dims(mask, axis=-1)

        env_mask = (targets == env_outputs).astype(int).astype(float)
        env_mask = jnp.expand_dims(env_mask, axis=-1)

        targets = util.get_action_1hot_batch(targets,
                                             n_actions)
        model_outputs = util.get_action_1hot_batch(model_outputs,
                                             n_actions)

        if obj_mask:
            meta_loss = -jnp.sum(mask * targets * jax.nn.log_softmax((rewarder_v), axis=-1), axis=-1).mean()
        else:
            meta_loss = -jnp.sum(targets * jax.nn.log_softmax((rewarder_v), axis=-1), axis=-1).mean()

        rewarder_loss = meta_loss 

        if cost_type == 0:
            incentive_cost = jnp.abs(env_mask * rewarder_v).mean()
            incentive_cost2 = jnp.abs((1 - env_mask) * rewarder_v).mean()
            rewarder_loss += (cost_reg_coeff * incentive_cost + cost_reg_coeff2 * incentive_cost2)
        else:
            incentive_cost = jnp.abs((model_v + rewarder_v).max(axis=-1) - 2 - model_v.max(axis=-1)).mean()
            rewarder_loss += (cost_reg_coeff * incentive_cost)

        return rewarder_loss, meta_loss, {'critic_info': critic_info,
                               'rewarder_loss': rewarder_loss,  \
                               'meta_loss': meta_loss, 'incentive_cost': incentive_cost}

    return meta_loss_fn()


def rewarder_loss_func(main_rewarder_params: FrozenDict,
                        rewarder: Model,
                        old_batches: Batch,
                        new_batches: Batch,
                        n_actions: int,
                        r_multiplier: float,
                        discount: float,
                        coop_qcritic: Model,
                        env_model_qcritic: Model,
                        env_model_vcritic: Model,
                        rewarder_model_qcritic: Model,
                        rewarder_model_vcritic: Model,
                        cost_reg_coeff: float,
                        cost_reg_coeff2: float,
                        cost_type: int,
                        obj_mask: bool):    

    rewarder_loss, meta_loss, info = loss_rewarder(main_rewarder_params,
                                                    rewarder,
                                                    old_batches,
                                                    new_batches,
                                                    n_actions,
                                                    r_multiplier,
                                                    discount,
                                                    coop_qcritic,
                                                    env_model_qcritic,
                                                    env_model_vcritic,
                                                    rewarder_model_qcritic,
                                                    rewarder_model_vcritic,
                                                    cost_reg_coeff,
                                                    cost_reg_coeff2,
                                                    cost_type,
                                                    obj_mask)
    return rewarder_loss, info


@jax.partial(jax.jit, static_argnums=(4, 5, 6, 12, 13, 14, 15))
def update_rewarder(main_rewarder_params: FrozenDict,
                    rewarder: Model,
                    old_batches: Batch,
                    new_batches: Batch,
                    n_actions: int,
                    r_multiplier: float,
                    discount: float,
                    coop_qcritic: Model,
                    env_model_qcritic: Model,
                    env_model_vcritic: Model,
                    rewarder_model_qcritic: Model,
                    rewarder_model_vcritic: Model,
                    cost_reg_coeff: float,
                    cost_reg_coeff2: float,
                    cost_type: int,
                    obj_mask: bool):                        

    grad_fn = jax.grad(rewarder_loss_func, has_aux=True, argnums=0)

    grads, info = grad_fn(
        main_rewarder_params,
        rewarder,
        old_batches,
        new_batches,
        n_actions,
        r_multiplier,
        discount,
        coop_qcritic,
        env_model_qcritic,
        env_model_vcritic,
        rewarder_model_qcritic,
        rewarder_model_vcritic,
        cost_reg_coeff,
        cost_reg_coeff2,
        cost_type,
        obj_mask)

    updates, new_rewarder_opt_state = rewarder.tx.update(grads,
                                                    rewarder.opt_state,
                                                    rewarder.params)

    new_rewarder_params = optax.apply_updates(rewarder.params, updates)

    new_rewarder = rewarder.replace(step=rewarder.step + 1,
                        params=new_rewarder_params,
                        opt_state=new_rewarder_opt_state)

    return new_rewarder, info


@jax.partial(jax.jit, static_argnums=(2))
def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_multimap(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)
