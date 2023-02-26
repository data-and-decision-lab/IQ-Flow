"""Actor critic with advantage function.

Advantage function is estimated by 1-step TD(0) error.
"""

from typing import Sequence, Tuple, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
import collections
import os

from iqflow.networks import nets
from iqflow.utils import utils as util

from iqflow.networks.common import InfoDict, Model, PRNGKey, Params


Batch = collections.namedtuple(
    'Batch',
    ['obs', 'obs_v', 'action', 'action_all', 
     'r_sampled', 'reward', 'obs_next', 'obs_v_next', 'done'])


@jax.partial(jax.jit, static_argnums=(4, 5, 7))
def _update_jit(
    actor: Model, critic: Model, target_critic: Model, batch: Batch,
    discount: float, tau: float, epsilon: float,
    entropy_coeff: float) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    new_critic, critic_info = update_critic(critic, target_critic,
                                            batch, discount)
    new_target_critic = target_update(new_critic, target_critic, tau)

    new_actor, actor_info = update_actor(actor, new_critic, epsilon,
                                         batch, discount, entropy_coeff)

    return new_actor, new_critic, new_target_critic, {
        **critic_info,
        **actor_info,
    }


class ActorCritic(object):

    def __init__(self, obs_space, config, dim_obs, n_actions, nn_config, agent_name,
                 agent_id, rng, obs_image_vec=False, l_obs=None):
        """Initialization.

        Args:
            config: ConfigDict
            dim_obs: list, if obs is an image; or an integer, if obs is a 1D vector
            n_actions: integer size of discrete action space
            nn: ConfigDict
            agent_name: string
            agent_id: integer
            obs_image_vec: if true, then agent has both image and 1D vector observation
            l_obs: integer size of 1D vector observation, used only if obs_image_vec
        """
        self.agent_id = agent_id
        self.agent_name = agent_name

        self.obs_space = obs_space
        self.dim_obs = dim_obs
        self.image_obs = isinstance(self.dim_obs, list)
        self.n_actions = n_actions
        self.nn = nn_config
        # -------------------
        # Used only when agent's observation has both image and 1D vector parts
        self.obs_image_vec = obs_image_vec
        self.l_obs = l_obs
        # -------------------
        
        self.entropy_coeff = config.entropy_coeff
        self.gamma = config.gamma
        self.lr_actor = config.lr_actor
        self.lr_v = config.lr_v
        self.tau = config.tau
        self.epsilon_start = config.epsilon_start

        self.rng = rng

        self.create_networks()

    def create_networks(self):

        rng, actor_key, critic_key = jax.random.split(self.rng, 3)

        if self.obs_image_vec:
            raise NotImplementedError
        elif self.image_obs:
            actor_def = nets.Actor(self.n_actions, self.nn.n_filters,
                                           self.nn.kernel, self.nn.stride,
                                           self.nn.n_h1, self.nn.n_h2)
            critic_def = nets.VNet(self.nn.n_filters,
                                           self.nn.kernel, self.nn.stride,
                                           self.nn.n_h1, self.nn.n_h2)
            observations = self.obs_space.sample()[np.newaxis]
            actor = Model.create(actor_def,
                             inputs=[actor_key, observations, self.epsilon_start],
                             tx=optax.adam(learning_rate=self.lr_actor))
            critic = Model.create(critic_def,
                              inputs=[critic_key, observations],
                              tx=optax.adam(learning_rate=self.lr_v))
            target_critic = Model.create(critic_def,
                              inputs=[critic_key, observations],
                              tx=optax.adam(learning_rate=self.lr_v))
        else:
            actor_def = nets.ActorMLP(self.n_actions, self.nn.n_h1, self.nn.n_h2)
            critic_def = nets.VNetMLP(self.nn.n_h1, self.nn.n_h2)
            observations = self.obs_space.sample()[np.newaxis]
            actor = Model.create(actor_def,
                                inputs=[actor_key, observations, self.epsilon_start],
                                tx=optax.adam(learning_rate=self.lr_actor))
            critic = Model.create(critic_def,
                              inputs=[critic_key, observations],
                              tx=optax.adam(learning_rate=self.lr_v))
            target_critic = Model.create(critic_def,
                              inputs=[critic_key, observations],
                              tx=optax.adam(learning_rate=self.lr_v))

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.rng = rng

    def run_actor(self, obs, epsilon, obs_v=None):

        rng, actions = nets.sample_actions(self.rng,
                                            self.actor.apply_fn,
                                            self.actor.params,
                                            obs,
                                            obs_v,
                                            epsilon)
        self.rng = rng

        return actions.item()

    def train(self, buf: Any, epsilon: float) -> InfoDict:

        batch = Batch(*[jnp.array(getattr(buf, name)) for name in Batch._fields])

        new_actor, new_critic, new_target_critic, info = _update_jit(
            self.actor, self.critic, self.target_critic, batch, self.gamma,
            self.tau, epsilon, self.entropy_coeff)

        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic

        return info


def update_critic(critic: Model, target_critic: Model, batch: Batch,
                 discount: float) -> Tuple[Model, InfoDict]:
    if len(batch.obs_v) > 0:
        v_target_next = target_critic(batch.obs_next, batch.obs_v_next)
    else:
        v_target_next = target_critic(batch.obs_next)

    td_target = batch.reward + discount * (1 - batch.done) * v_target_next.squeeze(1)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        if len(batch.obs_v) > 0:
            v = critic.apply({'params': critic_params}, batch.obs,
                              batch.obs_v)
        else:
            v = critic.apply({'params': critic_params}, batch.obs)
        critic_loss = ((v.squeeze(1) - td_target)**2).mean(0)
    
        return critic_loss, {
            'critic_loss': critic_loss,
            'v': v.mean(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


def update_actor(actor: Model, critic: Model, epsilon: float,
                 batch: Batch, discount: float, 
                 entropy_coeff: float) -> Tuple[Model, InfoDict]:

    if len(batch.obs_v) > 0:
        v = critic(batch.obs, batch.obs_v)
        v_next = critic(batch.obs_next, batch.obs_v_next)
    else:
        v = critic(batch.obs)
        v_next = critic(batch.obs_next)

    td_error = batch.reward.reshape(-1, 1) + discount * (1 - batch.done).reshape(-1, 1) * v_next - v
    actions_1hot = util.process_actions(batch.action, actor.apply_fn.n_actions)
    
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        if len(batch.obs_v) > 0:
            log_probs= actor.apply({'params': actor_params}, batch.obs, 
                                    batch.obs_v, epsilon, return_logits=False)
            probs = jnp.exp(log_probs)
        else:
            log_probs = actor.apply({'params': actor_params}, batch.obs, 
                                    epsilon, return_logits=False)
            probs = jnp.exp(log_probs)

        log_probs_taken = jnp.log(jnp.multiply(probs, actions_1hot).sum(axis=1) + 1e-15)
        
        entropy = -(probs * log_probs).sum(-1)
        policy_loss = -jnp.multiply(log_probs_taken, td_error.squeeze(-1))
        actor_loss = (policy_loss - entropy_coeff * entropy).sum(0)
        return actor_loss, {'actor_loss': actor_loss}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_multimap(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)
