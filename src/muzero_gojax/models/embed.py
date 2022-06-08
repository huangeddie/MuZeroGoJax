"""
Models that map Go states to other vector spaces, which can be used for feature extraction or
dimensionality reduction.
"""

import gojax
import haiku as hk
import jax.nn
import jax.numpy as jnp

from muzero_gojax.models import base


class Identity(base.BaseGoModel):
    """Identity model. Should be used with the real transition."""

    def __call__(self, states):
        return states


class BlackPerspective(base.BaseGoModel):
    """Converts all states whose turn is white to black's perspective."""

    def __call__(self, states):
        return jnp.where(jnp.expand_dims(gojax.get_turns(states), (1, 2, 3)),
                         gojax.swap_perspectives(states), states)


class BlackCNNLite(base.BaseGoModel):
    """Black perspective embedding followed by a light-weight CNN neural network."""

    def __init__(self, board_size, *args, **kwargs):
        super().__init__(board_size, *args, **kwargs)
        self._to_black = BlackPerspective(board_size, *args, **kwargs)
        self._conv1 = hk.Conv2D(32, (3, 3), data_format='NCHW')
        self._conv2 = hk.Conv2D(32, (3, 3), data_format='NCHW')

    def __call__(self, states):
        return self._conv2(jax.nn.relu(self._conv1(self._to_black(states).astype(float))))
