from typing import Sequence, Tuple, Callable, Any

from numpy.matrixlib.defmatrix import _convert_from_string

import flax.linen as nn
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from ml_collections import ConfigDict

from iqflow.networks.common import MLP, Params, PRNGKey, default_init, param_init, zero_init, mini_init


Array = Any


def abs_tanh(x: Array) -> Array:
  """Leaky rectified linear unit activation function.
  Args:
    x : input array
    negative_slope : array or scalar specifying the negative slope (default: 0.01)
  """
  return jnp.abs(jnp.tanh(x))


class ConvNet(nn.Module):
    feature: int
    kernel: Sequence[int]
    stride: Sequence[int]
    padding: str = "SAME"
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = obs.astype(jnp.float32)
        x = nn.Conv(self.feature,
                    kernel_size=self.kernel,
                    strides=self.stride,
                    kernel_init=self.kernel_init,
                    padding=self.padding)(x)
        x = nn.relu(x)
        if len(x.shape) == 4:
            x = x.reshape([x.shape[0], -1])
        else:
            x = x.reshape([-1])
        return x


class Actor(nn.Module):
    n_actions: int
    n_filters: int
    kernel: Sequence[int]
    stride: Sequence[int]
    n_h1: int
    n_h2: int
    epsilon: bool = False

    @nn.compact
    def __call__(self, 
                 obs: jnp.ndarray,
                 epsilon: float = None,
                 return_logits: bool = False) -> jnp.ndarray:
        conv_out = ConvNet(self.n_filters,
                           self.kernel,
                           self.stride)(obs)
        # print("conv out shape: ", conv_out.shape)
        # input()
        hiddens = (self.n_h1, self.n_h2, self.n_actions)
        # print("hiddens: ", hiddens)
        # input()
        outputs = MLP(hiddens)(conv_out)
        probs = nn.softmax(outputs)
        if self.epsilon:
            probs = (1 - epsilon) * probs + epsilon / self.n_actions
        log_probs = jnp.log(probs)
        if return_logits:
            return outputs, log_probs
        else:
            return log_probs


class ActorMLP(nn.Module):
    n_actions: int
    n_h1: int
    n_h2: int
    epsilon: bool = False

    @nn.compact
    def __call__(self, 
                 obs: jnp.ndarray,
                 epsilon: float = None,
                 return_logits: bool = False) -> jnp.ndarray:
        hiddens = (self.n_h1, self.n_h2, self.n_actions)
        outputs = MLP(hiddens)(obs)
        probs = nn.softmax(outputs)
        if self.epsilon:
            probs = (1 - epsilon) * probs + epsilon / self.n_actions
        log_probs = jnp.log(probs)
        if return_logits:
            return outputs, log_probs
        else:
            return log_probs


class ActorImageVec(nn.Module):
    n_actions: int
    n_filters: int
    kernel: Sequence
    stride: Sequence
    n_h1: int
    n_h2: int
    kernel_init: Callable = default_init()
    epsilon: bool = True

    @nn.compact
    def __call__(self,
                 obs_image: jnp.ndarray,
                 obs_vec: jnp.ndarray,
                 epsilon: float = None,
                 return_logits: bool = False) -> jnp.ndarray:
        conv_out = ConvNet(self.n_filters,
                           self.kernel,
                           self.stride)(obs_image)
        h1 = nn.Dense(self.n_h1,
                      kernel_init=self.kernel_init)(conv_out)
        h1 = nn.relu(h1)
        h1 = jnp.concatenate([h1, obs_vec], axis=1)
        h2 = nn.Dense(self.n_h2,
                      kernel_init=self.kernel_init)(h1)
        h2 = nn.relu(h2)
        outputs = nn.Dense(self.n_actions,
                      kernel_init=self.kernel_init)(h2)
        probs = nn.softmax(outputs)
        if self.epsilon:
            probs = (1 - epsilon) * probs + epsilon / self.n_actions
        log_probs = jnp.log(probs)
        if return_logits:
            return outputs, log_probs
        else:
            return log_probs


class Reward(nn.Module):
    n_filters: int
    kernel: Sequence
    stride: Sequence
    n_h1: int
    n_h2: int
    n_recipients: int
    n_actions: int
    output_nonlinearity: Callable = abs_tanh
    kernel_init: Callable = default_init()
    last_kernel_init: Callable = default_init()

    @nn.compact
    def __call__(self,
                 obs: jnp.ndarray,
                 a_others: jnp.ndarray, out: bool = False) -> jnp.ndarray:

        if len(obs.shape) == 4:
            obs = jnp.expand_dims(obs, axis=0)
        if len(a_others.shape) == 1:
            a_others = jnp.expand_dims(a_others, axis=0)
        b_s, a_s, h, w, c = obs.shape
        obs = obs.reshape(b_s * a_s, h, w, c)
        conv_out = ConvNet(self.n_filters,
                           self.kernel,
                           self.stride)(obs)
        conv_out = conv_out.reshape(b_s, -1)
        conv_reduced = nn.Dense(self.n_h1,
                      kernel_init=self.kernel_init)(conv_out)
        conv_reduced = nn.relu(conv_reduced)

        concated = jnp.concatenate([conv_reduced, a_others], axis=-1)

        h2 = nn.Dense(self.n_h2,
                      kernel_init=self.kernel_init)(concated)
        h2 = nn.relu(h2)

        reward_out = nn.Dense(self.n_recipients,
                    kernel_init=self.last_kernel_init)(h2)

        save_reward_out = reward_out

        if self.output_nonlinearity:
            reward_out = self.output_nonlinearity(reward_out)

        if b_s == 1:
            reward_out = reward_out.squeeze(0)

        if out:
            return reward_out, save_reward_out

        return reward_out


class RewardMLP(nn.Module):
    n_hr1: int
    n_hr2: int
    n_recipients: int
    n_actions: int
    output_nonlinearity: Callable = abs_tanh
    kernel_init: Callable = default_init()
    last_kernel_init: Callable = default_init()

    @nn.compact
    def __call__(self,
                 obs: jnp.ndarray,
                 a_others: jnp.ndarray, out: bool = False) -> jnp.ndarray:
        if len(obs.shape) == 2:
            obs = jnp.expand_dims(obs, axis=0)
        b_s, a_s, f_s = obs.shape
        obs = obs.reshape(b_s, -1)

        if len(a_others.shape) == 1:
            a_others = jnp.expand_dims(a_others, axis=0)

        concated = jnp.concatenate([obs, a_others], axis=-1)
        h1 = nn.Dense(self.n_hr1,
                      kernel_init=self.kernel_init)(concated)
        h1 = nn.relu(h1)
        h2 = nn.Dense(self.n_hr2,
                      kernel_init=self.kernel_init)(h1)
        h2 = nn.relu(h2)
        reward_out = nn.Dense(1 * self.n_recipients,
                      kernel_init=self.last_kernel_init)(h2)

        save_reward_out = reward_out

        if self.output_nonlinearity:
            reward_out = self.output_nonlinearity(reward_out)

        if out:
            return reward_out, save_reward_out

        return reward_out


class VNet(nn.Module):
    n_filters: int
    kernel: Sequence[int]
    stride: Sequence[int]
    n_h1: int
    n_h2: int
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        conv_out = ConvNet(self.n_filters,
                           self.kernel,
                           self.stride)(obs)
        hiddens = (self.n_h1, self.n_h2, 1)
        output = MLP(hiddens)(conv_out)
        return output


class QNet(nn.Module):
    n_filters: int
    kernel: Sequence[int]
    stride: Sequence[int]
    n_h1: int
    n_h2: int
    n_agents: int
    n_actions: int
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(self, obs: jnp.ndarray, all_actions: jnp.ndarray) -> jnp.ndarray:
        b_s, a_s, h, w, c = obs.shape
        obs = obs.reshape(b_s * a_s, h, w, c)
        conv_out = ConvNet(self.n_filters,
                           self.kernel,
                           self.stride)(obs)
        conv_out = conv_out.reshape(b_s, -1)
        hiddens = (self.n_h1, self.n_h2, self.n_agents * self.n_actions)
        concated = jnp.concatenate([conv_out, all_actions], axis=-1)
        output = MLP(hiddens)(concated)
        return output


class QNetNAgent(nn.Module):
    n_filters: int
    kernel: Sequence[int]
    stride: Sequence[int]
    n_h1: int
    n_h2: int
    n_agents: int
    n_actions: int
    kernel_init: Callable = default_init()

    def setup(self):
        hiddens = (self.n_h1, self.n_h2, self.n_actions)
        self.conv = ConvNet(self.n_filters,
                           self.kernel,
                           self.stride)
        self.mlp_list = [MLP(hiddens) for _ in range(self.n_agents)]


    def __call__(self, obs: jnp.ndarray, all_actions: jnp.ndarray) -> jnp.ndarray:
        b_s, a_s, h, w, c = obs.shape
        obs = obs.reshape(b_s * a_s, h, w, c)
        conv_out = self.conv(obs)
        conv_out = conv_out.reshape(b_s, -1)

        output_list = []
        
        for i, mlp in enumerate(self.mlp_list):
            other_act = all_actions.at[..., i*self.n_actions:(i+1)*self.n_actions].set(0.0)
            concat = jnp.concatenate([conv_out, other_act], axis=-1)
            out = mlp(concat)
            output_list.append(out)

        output = jnp.concatenate(output_list, axis=-1)
        return output


class CentralizedVNet(nn.Module):
    n_filters: int
    kernel: Sequence[int]
    stride: Sequence[int]
    n_h1: int
    n_h2: int
    n_agents: int
    n_actions: int
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        b_s, a_s, h, w, c = obs.shape
        obs = obs.reshape(b_s * a_s, h, w, c)
        conv_out = ConvNet(self.n_filters,
                           self.kernel,
                           self.stride)(obs)
        conv_out = conv_out.reshape(b_s, -1)
        hiddens = (self.n_h1, self.n_h2, 1)
        output = MLP(hiddens)(conv_out)
        return output


class CentralizedInterestVNet(nn.Module):
    n_filters: int
    kernel: Sequence[int]
    stride: Sequence[int]
    n_h1: int
    n_h2: int
    n_agents: int
    n_actions: int
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        b_s, a_s, h, w, c = obs.shape
        obs = obs.reshape(b_s * a_s, h, w, c)
        conv_out = ConvNet(self.n_filters,
                           self.kernel,
                           self.stride)(obs)
        conv_out = conv_out.reshape(b_s, -1)
        hiddens = (self.n_h1, self.n_h2, self.n_agents)
        output = MLP(hiddens)(conv_out)
        return output


class ModelQNet(nn.Module):
    n_filters: int
    kernel: Sequence[int]
    stride: Sequence[int]
    n_h1: int
    n_h2: int
    n_agents: int
    n_actions: int
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(self, obs: jnp.ndarray, all_actions: jnp.ndarray) -> jnp.ndarray:
        b_s, a_s, h, w, c = obs.shape
        obs = obs.reshape(b_s * a_s, h, w, c)
        conv_out = ConvNet(self.n_filters,
                           self.kernel,
                           self.stride)(obs)
        conv_out = conv_out.reshape(b_s, -1)
        hiddens = (self.n_h1, self.n_h2, self.n_agents * self.n_actions)
        concated = jnp.concatenate([conv_out, all_actions], axis=-1)
        output = MLP(hiddens)(concated)
        return output


class ModelQNetNAgent(nn.Module):
    n_filters: int
    kernel: Sequence[int]
    stride: Sequence[int]
    n_h1: int
    n_h2: int
    n_agents: int
    n_actions: int
    kernel_init: Callable = default_init()

    def setup(self):
        hiddens = (self.n_h1, self.n_h2, self.n_actions)
        self.conv = ConvNet(self.n_filters,
                           self.kernel,
                           self.stride)
        self.mlp_list = [MLP(hiddens) for _ in range(self.n_agents)]


    def __call__(self, obs: jnp.ndarray, all_actions: jnp.ndarray) -> jnp.ndarray:
        b_s, a_s, h, w, c = obs.shape
        obs = obs.reshape(b_s * a_s, h, w, c)
        conv_out = self.conv(obs)
        conv_out = conv_out.reshape(b_s, -1)

        output_list = []
        
        for i, mlp in enumerate(self.mlp_list):
            other_act = all_actions.at[..., i*self.n_actions:(i+1)*self.n_actions].set(0.0)
            concat = jnp.concatenate([conv_out, other_act], axis=-1)
            out = mlp(concat)
            output_list.append(out)

        output = jnp.concatenate(output_list, axis=-1)
        return output


class VNetMLP(nn.Module):
    n_h1: int
    n_h2: int
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        hiddens = (self.n_h1, self.n_h2, 1)
        output = MLP(hiddens)(obs)
        return output


class QNetMLP(nn.Module):
    n_h1: int
    n_h2: int
    n_agents: int
    n_actions: int
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(self, obs: jnp.ndarray, all_actions: jnp.ndarray) -> jnp.ndarray:
        b_s, a_s, f_s = obs.shape
        obs = obs.reshape(b_s, a_s * f_s)
        hiddens = (self.n_h1, self.n_h2, self.n_agents * self.n_actions)
        concated = jnp.concatenate([obs, all_actions], axis=-1)
        output = MLP(hiddens)(concated)
        return output


class QNetMLPNAgent(nn.Module):
    n_h1: int
    n_h2: int
    n_agents: int
    n_actions: int
    kernel_init: Callable = default_init()

    def setup(self):
        hiddens = (self.n_h1, self.n_h2, self.n_actions)
        self.mlp_list = [MLP(hiddens) for _ in range(self.n_agents)]

    
    def __call__(self, obs: jnp.ndarray, all_actions: jnp.ndarray) -> jnp.ndarray:
        b_s, a_s, f_s = obs.shape
        obs = obs.reshape(b_s, a_s * f_s)

        output_list = []
        
        for i, mlp in enumerate(self.mlp_list):
            other_act = all_actions.at[..., i*self.n_actions:(i+1)*self.n_actions].set(0.0)
            concat = jnp.concatenate([obs, other_act], axis=-1)
            out = mlp(concat)
            output_list.append(out)

        output = jnp.concatenate(output_list, axis=-1)
        return output


class CentralizedVNetMLP(nn.Module):
    n_h1: int
    n_h2: int
    n_agents: int
    n_actions: int
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        b_s, a_s, f_s = obs.shape
        obs = obs.reshape(b_s, a_s * f_s)
        hiddens = (self.n_h1, self.n_h2, 1)
        output = MLP(hiddens)(obs)
        return output


class CentralizedInterestVNetMLP(nn.Module):
    n_h1: int
    n_h2: int
    n_agents: int
    n_actions: int
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        b_s, a_s, f_s = obs.shape
        obs = obs.reshape(b_s, a_s * f_s)
        hiddens = (self.n_h1, self.n_h2, self.n_agents)
        output = MLP(hiddens)(hiddens)
        return output


class ModelQNetMLP(nn.Module):
    n_h1: int
    n_h2: int
    n_agents: int
    n_actions: int
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(self, obs: jnp.ndarray, all_actions: jnp.ndarray) -> jnp.ndarray:
        b_s, a_s, f_s = obs.shape
        obs = obs.reshape(b_s, a_s * f_s)
        hiddens = (self.n_h1, self.n_h2, self.n_agents * self.n_actions)
        concated = jnp.concatenate([obs, all_actions], axis=-1)
        output = MLP(hiddens)(concated)
        return output


class ModelQNetMLPNAgent(nn.Module):
    n_h1: int
    n_h2: int
    n_agents: int
    n_actions: int
    kernel_init: Callable = default_init()

    def setup(self):
        hiddens = (self.n_h1, self.n_h2, self.n_actions)
        self.mlp_list = [MLP(hiddens) for _ in range(self.n_agents)]

    
    def __call__(self, obs: jnp.ndarray, all_actions: jnp.ndarray) -> jnp.ndarray:
        b_s, a_s, f_s = obs.shape
        obs = obs.reshape(b_s, a_s * f_s)

        output_list = []
        
        for i, mlp in enumerate(self.mlp_list):
            other_act = all_actions.at[..., i*self.n_actions:(i+1)*self.n_actions].set(0.0)
            concat = jnp.concatenate([obs, other_act], axis=-1)
            out = mlp(concat)
            output_list.append(out)

        output = jnp.concatenate(output_list, axis=-1)
        return output


class VNetImageVec(nn.Module):
    n_filters: int
    kernel: Sequence[int]
    stride: Sequence[int]
    n_h1: int
    n_h2: int
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(self, 
                 obs_image: jnp.ndarray,
                 obs_vec: jnp.ndarray) -> jnp.ndarray:
        conv_out = ConvNet(self.n_filters,
                           self.kernel,
                           self.stride)(obs_image)
        h1 = nn.Dense(self.n_h1,
                      kernel_init=self.kernel_init)(conv_out)
        h1 = nn.relu(h1)
        h1 = jnp.concatenate([h1, obs_vec], axis=1)
        h2 = nn.Dense(self.n_h2,
                      kernel_init=self.kernel_init)(h1)
        h2 = nn.relu(h2)
        output = nn.Dense(1,
                      kernel_init=self.kernel_init)(h2)
        return output


@jax.partial(jax.jit, static_argnums=(1,))
def sample_actions(
        rng: PRNGKey,
        actor_def: nn.Module,
        actor_params: Params,
        observations: jnp.ndarray,
        observations_vec: jnp.ndarray = None,
        epsilon: float = 0.0) -> Tuple[PRNGKey, jnp.ndarray]:

    if observations_vec:
        log_probs = actor_def.apply({'params': actor_params}, observations,
                            observations_vec, epsilon)
    else:
        log_probs = actor_def.apply({'params': actor_params}, observations,
                            epsilon)
                            
    rng, key = jax.random.split(rng)
    actions = jax.random.categorical(key, log_probs)
    
    return rng, actions


@jax.partial(jax.jit, static_argnums=(1,))
def sample_actions_pg(
        rng: PRNGKey,
        actor_def: nn.Module,
        actor_params: Params,
        observations: jnp.ndarray,
        observations_vec: jnp.ndarray = None) -> Tuple[PRNGKey, jnp.ndarray]:

    if observations_vec:
        log_probs = actor_def.apply({'params': actor_params}, observations,
                            observations_vec)
    else:
        log_probs = actor_def.apply({'params': actor_params}, observations)
                            
    rng, key = jax.random.split(rng)
    actions = jax.random.categorical(key, log_probs)
    
    return rng, actions