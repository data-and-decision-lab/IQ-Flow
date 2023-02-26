# from https://github.com/ikostrikov/jaxrl

import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from jax._src.util import prod
from flax.training import train_state


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


def mini_init(scale: Optional[float] = jnp.sqrt(2), 
              column_axis=-1, dtype=jnp.float_):
    # Randomly initializes parameters to approximately 0
    def init(key, shape, dtype=dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        if len(shape) < 2:
            raise ValueError("orthogonal initializer requires at least a 2D shape")
        n_rows, n_cols = prod(shape) // shape[column_axis], shape[column_axis]
        matrix_shape = (n_cols, n_rows) if n_rows < n_cols else (n_rows, n_cols)
        A = jax.random.normal(key, matrix_shape, dtype)
        Q, R = jnp.linalg.qr(A)
        diag_sign = jax.lax.broadcast_to_rank(jnp.sign(jnp.diag(R)), rank=Q.ndim)
        Q *= diag_sign # needed for a uniform distribution
        if n_rows < n_cols: Q = Q.T
        Q = jnp.reshape(Q, tuple(np.delete(shape, column_axis)) + (shape[column_axis],))
        Q = jnp.moveaxis(Q, -1, column_axis)
        return scale * Q / 1000

    return init


def param_init(scale: Optional[float] = jnp.sqrt(2), 
              column_axis=-1, dtype=jnp.float_):
    def init(key, shape, dtype=dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        return jax.random.normal(key, shape, dtype)

    return init


def zero_init(key, shape, dtype=jnp.float_):
    return nn.initializers.zeros(key, shape, dtype=dtype)


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x


# TODO: Replace with TrainState when it's ready
# https://github.com/google/flax/blob/master/docs/flip/1009-optimizer-api.md#train-state
# @flax.struct.dataclass
class Model(train_state.TrainState):
    step: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':
        variables = model_def.init(*inputs)

        _, params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({'params': self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, 'Model']:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state=new_opt_state), info

    def apply_gradient_params(self, loss_fn, clip_grad=False) -> Tuple[Any, 'Model']:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)

        if clip_grad:
            grads = self.grad_clip(grads)

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        return new_params, info


class Model2Optim(train_state.TrainState):
    step: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx1: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    tx2: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state1: Optional[optax.OptState] = None
    opt_state2: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx1: Optional[optax.GradientTransformation] = None,
               tx2: Optional[optax.GradientTransformation] = None) -> 'Model':
        variables = model_def.init(*inputs)

        _, params = variables.pop('params')

        if tx1 is not None:
            opt_state1 = tx1.init(params)
        else:
            opt_state1 = None

        if tx2 is not None:
            opt_state2 = tx2.init(params)
        else:
            opt_state2 = None

        return cls(step=1,
                   apply_fn=model_def,
                   params=params,
                   tx=tx1,
                   opt_state=opt_state1,
                   tx1=tx1,
                   tx2=tx2,
                   opt_state1=opt_state1,
                   opt_state2=opt_state2)

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({'params': self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient1(self, loss_fn) -> Tuple[Any, 'Model']:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)

        updates, new_opt_state = self.tx1.update(grads, self.opt_state1,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state1=new_opt_state), info

    def apply_gradient2(self, loss_fn) -> Tuple[Any, 'Model']:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)

        updates, new_opt_state = self.tx2.update(grads, self.opt_state2,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state2=new_opt_state), info
