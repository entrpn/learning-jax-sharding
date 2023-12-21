# https://arxiv.org/pdf/2105.04663.pdf
# Section 5.1, Figure 7

import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'

import functools
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
from jax.sharding import PartitionSpec, NamedSharding
from jax.sharding import Mesh

from flax import linen as nn
from flax.training import train_state
from flax.linen import partitioning as nn_partitioning

import optax

# Suppose the input to such layers is tensor shape (B,S,M)
# B = batch_size
# S = sequence_length
# M = features
# Attention can be described as:
# y = Attention(Wq * x, Wk * x, Wv * x) * Wo

# Each Wq, Wk, Wv is a weight matrix that projects x into a 
# tensor of shape (B,S,N,D). 
# N = attention_heads

# The Wo weight matrix projects the attention result back to shape (B,S,M)

# The feed-forwrd layer can be described as:
# y = Relu(Win * x) * Wout
# Where Xin is a weight matrix that projects x into a
# tensor of shape (B,S,H).
# The Wout weight matrix projects the result back to shape (B,S,M)

class FlaxAttention(nn.Module):
  query_dim: int
  heads: int = 8
  dim_head: int = 64
  dropout: float = 0.0
  dtype: jnp.dtype = jnp.bfloat16

  def setup(self):
    self.inner_dim = self.dim_head * self.heads
    self.scale = self.dim_head ** -0.5
  
  @nn.compact
  def __call__(self, hidden_states, context=None, deterministic=True):
    context = hidden_states if context is None else context

    # Wq, Wk, Wv
    # Shape : MND
    # Shardings: X,Y,_
    query_proj = nn.Dense(
        self.inner_dim,
        kernel_init=nn.with_logical_partitioning(
          nn.initializers.lecun_normal(),
          ('embed','kv')), # Is this correct? hidden should be renamed to something else
        use_bias=False,
        dtype=self.dtype,
        name="to_q"
    )(context)

    print("context.shape: ", context.shape)
    print("query_proj.shape: ", query_proj.shape)
    return query_proj
# 2D finalized

key = jax.random.key(0)

B = 8
S = 256
M = 640
x = jax.random.normal(key, (B,S,M))

# Create mesh
device_mesh = mesh_utils.create_device_mesh((2, 2))
mesh = Mesh(devices=device_mesh, axis_names=('data','model'))

def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
  return NamedSharding(mesh, pspec)
# Data sharding
x_sharding = mesh_sharding(PartitionSpec('data', 'model'))
x = jax.device_put(x, x_sharding)
print("Visualize x[0]: ")
jax.debug.visualize_array_sharding(x[0])
x_0 = x.device_buffers[0]
print("x[0] shape: ", x_0.shape)

attention = FlaxAttention(M)

def init_fn(k, x, model, optimizer):
  variables = model.init(k, x)
  state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=optimizer
  )
  return state

init_rngs = {'params' : jax.random.PRNGKey(1), 'dropout' : jax.random.PRNGKey(2)}
optimizer = optax.adam(learning_rate=0.001)

rules = (('batch', 'data'),
         ('embed', 'data'),
         #('kv', 'model'),
         ('hidden', 'model'))
logical_abstract_variables = jax.eval_shape(functools.partial(init_fn, model=attention, optimizer=optimizer), init_rngs, x)
logical_state_spec = nn.get_partition_spec(logical_abstract_variables)
logical_state_sharding = nn.logical_to_mesh_sharding(logical_state_spec, mesh, rules)
jit_init_fn = jax.jit(init_fn, static_argnums=(2,3),
                      in_shardings=(mesh_sharding(None), x_sharding),
                      out_shardings=logical_state_sharding)

initialized_state = jit_init_fn(init_rngs,x,attention, optimizer)
print("Visualize Wq sharding:")
to_q = initialized_state.params['to_q']['kernel'].value
jax.debug.visualize_array_sharding(to_q)

to_q_0 = to_q.device_buffers[0]
print("Wq shape: ", to_q.shape)
print("Wq_0 shape: ", to_q_0.shape)
print("x[0] shape: ", x_0.shape)

@functools.partial(jax.jit, in_shardings=(logical_state_sharding, x_sharding),
                   out_shardings=logical_state_sharding)
def train_step(state, x):
  def loss_unrolled(params):
    y = attention.apply({'params' : params}, x)
    return y.sum()
  grad_fn = jax.grad(loss_unrolled)
  grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state
# def apply_fn(state, x):
#   return attention.apply({"params: ": state.params}, x)

with mesh, nn_partitioning.axis_rules(rules):
  new_state = train_step(initialized_state, x)

# In this sharding configuration, we have x[0] with shape (4, 128, 640)
# While Wq, Wk, Wv, we have Wq[0], Wk[0], Wv[0] with shape (320, 256)
# The expected multiply per partition, if we did x[0] first row by Wq[0] first column:
# x[0][0] is 640 columns to Wq[0][0] is 320 rows. 
# x[0][0] is 128 rows to Wq[0][0] is 256 columns.
# if  activation, y[0][0] = x[0][0] * Wq[0][0], we should expect y[0][0] to be 128 x 320 

@functools.partial(jax.jit, in_shardings=(logical_state_sharding, x_sharding),
                   out_shardings=x_sharding)
def apply_fn(state, x):
  return state.apply_fn({"params" : state.params}, x)

with mesh, nn_partitioning.axis_rules(rules):
  y = apply_fn(new_state, x)

import pdb;pdb.set_trace()
#params['to_q']['kernel'].value