# https://arxiv.org/pdf/2105.04663.pdf
# Section 5.1, Figure 7

import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

from flax import linen as nn

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
      ('features', 'heads', None)
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
            ('','')
        ),
        dtype=self.dtype,
        name="to_out_0")
    self.dropout_layer = nn.Dropout(rate=self.dropout)
  
  def __call__(self, hidden_states, context=None, deterministic=True):
    context = hidden_states if context is None else context
    query_proj = self.query(hidden_states)
    key_proj = self.key(context)
    value_proj = self.value(context)

    b = hidden_states.shape[0]
    query_states = jnp.reshape(query_proj, (b, -1, self.heads, self.dim_head))
    key_states = jnp.reshape(key_proj, (b, -1, self.heads, self.dim_head))
    value_states = jnp.reshape(value_proj, (b, -1, self.heads, self.dim_head))

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

    hidden_states = self.proj_attn(hidden_states)

    return self.dropout_layer(hidden_states, deterministic=deterministic)

# 2D finalized



key = jax.random.key(0)

B = 8
S = 256
M = 40
x = jax.random.normal(key, (B,S,M))

attention = FlaxAttention(640)
init_rngs = {'params' : jax.random.PRNGKey(1), 'dropout' : jax.random.PRNGKey(2)}
variables = attention.init(init_rngs, x)
params = variables['params']
import pdb; pdb.set_trace()