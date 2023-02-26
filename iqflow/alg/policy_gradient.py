# Policy Gradient Algorithm

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


# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "False"
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"


Batch = collections.namedtuple(
    'Batch',
    ['obs', 'obs_v', 'action', 'action_all', 
     'r_sampled', 'reward', 'obs_next', 'obs_v_next', 'done'])


@jax.partial(jax.jit, static_argnums=(2, 3))
def _update_jit(
    actor: Model, batch: Batch,
    discount: float, 
    entropy_coeff: float) -> Tuple[Model, InfoDict]:

    new_actor, actor_info = update_actor(actor, 
                                         batch, discount, entropy_coeff)

    return new_actor, actor_info


class PolicyGradient(object):

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
        self.config = config

        self.rng = rng

        self.create_networks()

    def create_networks(self):

        rng, actor_key = jax.random.split(self.rng, 2)

        if self.obs_image_vec:
            raise NotImplementedError

        elif self.image_obs:
            actor_def = nets.Actor(self.n_actions, self.nn.n_filters,
                                           self.nn.kernel, self.nn.stride,
                                           self.nn.n_h1, self.nn.n_h2, epsilon=False)
            observations = self.obs_space.sample()[np.newaxis]
        else:
            actor_def = nets.ActorMLP(self.n_actions, self.nn.n_h1, self.nn.n_h2, epsilon=False)
            observations = self.obs_space.sample()[np.newaxis]

        if 'optimizer' in self.config and self.config.optimizer == 'adam':
            actor = Model.create(actor_def,
                                inputs=[actor_key, observations],
                                tx=optax.adam(learning_rate=self.lr_actor))
        else:
            actor = Model.create(actor_def,
                                inputs=[actor_key, observations],
                                tx=optax.sgd(learning_rate=self.lr_actor))

        self.actor = actor
        self.rng = rng

    def run_actor(self, obs, epsilon=None, obs_v=None):

        rng, actions = nets.sample_actions_pg(self.rng,
                                            self.actor.apply_fn,
                                            self.actor.params,
                                            obs,
                                            obs_v)
        self.rng = rng

        return actions.item()

    def train(self, buf: Any, epsilon=None) -> InfoDict:

        batch = Batch(*[jnp.array(getattr(buf, name)) for name in Batch._fields])

        new_actor, info = _update_jit(
            self.actor, batch, self.gamma,
            self.entropy_coeff)

        self.actor = new_actor

        return info


def update_actor(actor: Model,
                 batch: Batch, discount: float, 
                 entropy_coeff: float) -> Tuple[Model, InfoDict]:

    return_discount = jnp.cumprod(jnp.ones_like(batch.reward) * discount) / discount
    returns = jnp.multiply(batch.reward, return_discount)

    actions_1hot = util.process_actions(batch.action, actor.apply_fn.n_actions)
    
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        if len(batch.obs_v) > 0:
            log_probs= actor.apply({'params': actor_params}, batch.obs, 
                                    batch.obs_v, return_logits=False)
            probs = jnp.exp(log_probs)
        else:
            log_probs = actor.apply({'params': actor_params}, batch.obs, 
                                    return_logits=False)
            probs = jnp.exp(log_probs)

        log_probs_taken = jnp.log(jnp.multiply(probs, actions_1hot).sum(axis=1) + 1e-15)
        
        entropy = -(probs * log_probs).sum(-1)
        policy_loss = -jnp.multiply(log_probs_taken, returns.reshape(*log_probs_taken.shape))
        actor_loss = (policy_loss - entropy_coeff * entropy).sum(0)
        return actor_loss, {'actor_loss': actor_loss}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
