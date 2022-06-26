"""
Models that map Go states to other vector spaces, which can be used for feature extraction or
dimensionality reduction.
"""

import gojax
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._to_black = BlackPerspective(*args, **kwargs)
        self._simple_conv_block = base.SimpleConvBlock(hdim=32, odim=32, **kwargs)

    def __call__(self, states):
        return self._simple_conv_block(self._to_black(states))
