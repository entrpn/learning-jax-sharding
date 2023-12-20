# https://arxiv.org/pdf/2105.04663.pdf
# 3.2 Examples of expressiong in-operator parallelism
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import numpy as np
import jax
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
import jax.numpy as jnp

# Using Einstein Summation, Einsum to explain sharding. 
# The equivalient operation in XLA is Dot.

# In generalized matrix multiply, An Einsum can be expressed as:
# ABC, ACD -> ABD

# batch
A = 8
# arr_A outer dim
B = 4
# inner dim
C = 16
# arr_B outer dim
D = 4

key = jax.random.PRNGKey(0)
arr_A = jax.random.normal(key, (A, B, C))
arr_B = jax.random.normal(key, (A, C, D))

arr_C = jnp.einsum("ABC,ACD->ABD", arr_A, arr_B)

assert arr_C.shape == (A,B,D)
print("arr_C.shape: ", arr_C.shape)

# With GSPMD, the user can annotate operands and output will combine different
# parallelism modes. For a typical fully connected projection layer:
# BD, DF -> BF
# User can combine data and model parallelism by annotating:
# bd = mesh_split(bd, mesh, [0, -1])
# df = mesh_split(df, mesh, [-1, 1])

sharding = PositionalSharding(mesh_utils.create_device_mesh((2,4)))
A = jax.random.normal(key, (4, 16))
B = jax.random.normal(key, (16, 4))

A = jax.device_put(A, sharding.replicate(axis=1, keepdims=True))
print("A:")
jax.debug.visualize_array_sharding(A)
B = jax.device_put(B, sharding.replicate(axis=0, keepdims=True))
jax.debug.visualize_array_sharding(B)

C = jax.lax.dot(A,B)
print("C: ")
jax.debug.visualize_array_sharding(C)

C_0 = C.addressable_shards[0].data
assert C_0.shape == (2, 1)
print("C_0.shape: ", C_0.shape)
print("All visualizations should look like Figure 3 in the GSPMD paper but with a 2 x 4 instead of a 2 x 2 mesh.")

