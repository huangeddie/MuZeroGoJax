"""Go constants."""

import jax.numpy as jnp

BLACKS_TURN = False
WHITES_TURN = True

BLACK_CHANNEL_INDEX = 0
WHITE_CHANNEL_INDEX = 1
TURN_CHANNEL_INDEX = 2
KILLED_CHANNEL_INDEX = 3
PASS_CHANNEL_INDEX = 4
END_CHANNEL_INDEX = 5

NUM_CHANNELS = 6

# A kernel (OIHW format) used to in convolution used to expand a batch of 2D boolean arrays in
# all four cardinal directions. Temporarily returns a float array until CUDNN can support
# convolutions with booleans or integers.
CARDINALLY_CONNECTED_KERNEL = jnp.array(
    [[[[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]]]], dtype='bfloat16')
