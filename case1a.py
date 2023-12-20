# https://irhum.github.io/blog/pjit/#case-1a-mesh-axes-match
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import numpy as np
import jax
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

print("""
This is the case for AllGather.
      A: (full, sharded)
      B: (sharded, full)
      """)

sharding = PositionalSharding(mesh_utils.create_device_mesh((2,4)))
key = jax.random.PRNGKey(0)
A = jax.random.normal(key, (4, 16))
B = jax.random.normal(key, (16, 4))

# Case1A - We have a 2 x 4 mesh, where 2 is the X axis and 4 is the Y axis of the mesh.
# One way to partition A and B onto this mesh is by
# sharding the inner axis of both A and B and replicating the outer axis.

A = jax.device_put(A, sharding.replicate(axis=0, keepdims=True))
print("A:")
jax.debug.visualize_array_sharding(A)

# In here we must reshape the sharding to a 4 x 2 to make sure B is sharded properly
# across the X dimension while replicating across the Y dimension.
B = jax.device_put(B, sharding.reshape(4,2).replicate(axis=1, keepdims=True))
print("B:")
jax.debug.visualize_array_sharding(B)

# Validate 
A_0 = np.array(A.device_buffers[0])
assert A_0.shape == (4,4)
print("A_0.shape: ",A_0.shape)

A_4 = np.array(A.device_buffers[4])
print("Are A_0 and A_4 equal? ", (np.array_equal(A_0, A_4)))

B_0 = np.array(B.device_buffers[0])
assert B_0.shape == (4,4)
B_1 = np.array(B.device_buffers[1])
print("B_0.shape: ", B_0.shape)
print("Are B_0 and B_4 equal? ", (np.array_equal(B_0, B_1)))

# Now each device mutiplies the shard of A and B to produce part of the multiplied value of C.
C = jax.lax.dot(A,B)
print("C:")
jax.debug.visualize_array_sharding(C)

C_0 = np.array(C.device_buffers[0])
C_1 = np.array(C.device_buffers[1])
C_4 = np.array(C.device_buffers[4])

# All devices pull their necessary values from its neighbors and adds them to their
# own calculations. This is AllReduce.
print("All reduce happens...")
print("Are C_0 and C_1 equal? ", (np.array_equal(C_0, C_1)))
print("Are C_0 and C_4 equal? ", (np.array_equal(C_0, C_4)))
print("Are C_0 and C equal? ", (np.array_equal(C_0, C)))


# Note that there is duplication over the X axis, which results in both duplicate memory and computation