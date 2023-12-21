# https://arxiv.org/pdf/2105.04663.pdf
# Section 5.1, Figure 7

import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'

import functools
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec, NamedSharding
from jax.sharding import Mesh

from flax import linen as nn
from flax.training import train_state
from flax.linen import partitioning as nn_partitioning

import optax

import time

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
    inner_dim = self.dim_head * self.heads
    self.scale = self.dim_head ** -0.5

    # Wq, Wk, Wv
    # Shape : MND
    # Shardings: X,Y,_
    qkv_init_kernel = nn.with_logical_partitioning(
      nn.initializers.lecun_normal(),
      ('embed','heads')
    )

    self.query = nn.Dense(
        inner_dim,
        kernel_init=qkv_init_kernel,
        use_bias=False,
        dtype=self.dtype,
        name="to_q"
    )

    self.key = nn.Dense(
        inner_dim,
        kernel_init=qkv_init_kernel,
        use_bias=False,
        dtype=self.dtype,
        name="to_k"
    )

    self.value = nn.Dense(
        inner_dim,
        kernel_init=qkv_init_kernel,
        use_bias=False,
        dtype=self.dtype,
        name="to_v")
    self.proj_attn = nn.Dense(
        self.query_dim,
        kernel_init=nn.with_logical_partitioning(
            nn.initializers.lecun_normal(),
            ('heads','embed')
        ),
        dtype=self.dtype,
        name="to_out_0")
    self.dropout_layer = nn.Dropout(rate=self.dropout)
  
  def __call__(self, hidden_states, context=None, deterministic=True):
    context = hidden_states if context is None else context
    print("context.shape: ", context.shape)
    query_proj = self.query(hidden_states)
    print("query_proj.shape: ", query_proj.shape)
    key_proj = self.key(context)
    value_proj = self.value(context)

    # proj dims will be (batch, embed, heads * head_dim)
    # Reshaping below should not have any device movement since
    # heads is replicated.
    # Applying sharding constraint should be as follows:
    query_proj = nn.with_logical_constraint(query_proj, ('batch', 'embed', None))
    key_proj = nn.with_logical_constraint(key_proj, ('batch', 'embed', None))
    value_proj = nn.with_logical_constraint(value_proj, ('batch', 'embed', None))

    b = hidden_states.shape[0]
    query_states = jnp.reshape(query_proj, (b, -1, self.heads, self.dim_head))
    key_states = jnp.reshape(key_proj, (b, -1, self.heads, self.dim_head))
    value_states = jnp.reshape(value_proj, (b, -1, self.heads, self.dim_head))

    query_states = nn.with_logical_constraint(query_states, ('batch', 'embed', None, None))
    key_states = nn.with_logical_constraint(key_states, ('batch', 'embed', None, None))
    value_states = nn.with_logical_constraint(value_states, ('batch', 'embed', None, None))

    print("query_states.shape: ", query_states.shape)

    # Attn stability
    query_states = jnp.float32(query_states)
    key_states = jnp.float32(key_states)

    # compute attentions
    attention_scores = jnp.einsum("b t n h, b f n h -> b n f t", key_states, query_states)
    attention_scores = attention_scores * self.scale
    attention_probs = nn.softmax(attention_scores, axis=-1)

    # back to original dtype
    attention_probs = jnp.asarray(attention_probs, dtype=self.dtype)

    # attend to values
    hidden_states = jnp.einsum("b n f t, b t n h -> b f n h", attention_probs, value_states)
    b = hidden_states.shape[0]
    hidden_states = jnp.reshape(hidden_states, (b, -1, self.heads * self.dim_head))

    hidden_states = nn.with_logical_constraint(hidden_states, ('batch', 'kv', 'heads'))

    hidden_states = self.proj_attn(hidden_states)

    hidden_states = nn.with_logical_constraint(hidden_states,('batch', 'embed'))

    return self.dropout_layer(hidden_states, deterministic=deterministic)

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
# assert x_0.shape == (4,64,640)

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

rules = (
  ('batch', 'data'),
  ('embed', 'model'),
  ('hidden', 'model'),
)

logical_abstract_variables = jax.eval_shape(functools.partial(init_fn, model=attention, optimizer=optimizer), init_rngs, x)
logical_state_spec = nn.get_partition_spec(logical_abstract_variables)
logical_state_sharding = nn.logical_to_mesh_sharding(logical_state_spec, mesh, rules)
jit_init_fn = jax.jit(init_fn, static_argnums=(2,3),
                      in_shardings=(mesh_sharding(None), x_sharding),
                      out_shardings=logical_state_sharding)

initialized_state = jit_init_fn(init_rngs,x,attention, optimizer)
print("Visualize Wq sharding:")
jax.debug.visualize_array_sharding(initialized_state.params['to_q']['kernel'].value)
to_q = initialized_state.params['to_q']['kernel'].value
to_q_0 = to_q.device_buffers[0]
print("Wq shape: ", )
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
  s = time.time()
  for i in range(10):
    y = apply_fn(new_state, x)
  print("time for 10 itters: ", (time.time() - s))
