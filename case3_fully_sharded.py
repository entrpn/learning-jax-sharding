# https://irhum.github.io/blog/pjit/#full-sharding
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import numpy as np
import jax
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

sharding = PositionalSharding(mesh_utils.create_device_mesh((2,4)))
key = jax.random.PRNGKey(0)
A = jax.random.normal(key, (4, 16))
B = jax.random.normal(key, (16, 4))

print("""
      A: (sharded_X, sharded_Y)
      B: (sharded_X, sharded_Y)
      """)

# Case1A - We have a 2 x 4 mesh, where 2 is the X axis and 4 is the Y axis of the mesh.
# One way to partition A and B onto this mesh is by
# sharding the inner axis of both A and B and replicating the outer axis.

A = jax.device_put(A, sharding)
print("visualize A: ")
jax.debug.visualize_array_sharding(A)

# In here we must reshape the sharding to a 4 x 2 to make sure B is sharded properly
# across the X dimension while replicating across the Y dimension.
B = jax.device_put(B, sharding)
print("visualize B: ")
jax.debug.visualize_array_sharding(B)

# Validate 
A_0 = np.array(A.device_buffers[0])
assert A_0.shape == (2,4)
print("A_0.shape: ",A_0.shape)
A_4 = np.array(A.device_buffers[4])
print("Are A_0 and A_4 NOT equal? ", (np.array_equal(A_0, A_4)))
B_0 = np.array(B.device_buffers[0])
B_1 = np.array(B.device_buffers[1])
assert B_0.shape == (8,1)
print("B_0.shape: ", B_0.shape)
print("Are B_0 and B_4 equal? ", (np.array_equal(B_0, B_1)))

# Now each device mutiplies the shard of A and B to produce part of the multiplied value of C.
C = jax.lax.dot(A,B)
print("visualize C:")
jax.debug.visualize_array_sharding(C)
print("C.shape: ", C.shape)
C_0 = np.array(C.device_buffers[0])
print("C_0.shape: ", C_0.shape)
assert C_0.shape == (2, 1)
C_1 = np.array(C.device_buffers[1])
C_4 = np.array(C.device_buffers[4])

# In this case, all devices have a different part of C.
print("All gather happens...")
print("Are C_0 and C_1 NOT equal? ", (np.array_equal(C_0, C_1)))
print("Are C_0 and C_4 NOT equal? ", (np.array_equal(C_0, C_4)))
print("Are C_0 and C NOT equal? ", (np.array_equal(C_0, C)))
import pdb; pdb.set_trace()