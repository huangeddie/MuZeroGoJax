"""
Models that map Go states to other vector spaces, which can be used for feature extraction or
dimensionality reduction.
"""

import gojax
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._to_black = BlackPerspective(*args, **kwargs)
        self._simple_conv_block = base.SimpleConvBlock(hdim=self.hdim, odim=self.hdim, **kwargs)

    def __call__(self, states):
        return jax.nn.relu(self._simple_conv_block(self._to_black(states)))


class BlackCNNIntermediate(base.BaseGoModel):
    """Black perspective embedding followed by an intermediate CNN neural network."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._to_black = BlackPerspective(*args, **kwargs)
        self._conv_block_1 = base.SimpleConvBlock(hdim=self.hdim, odim=self.hdim, **kwargs)
        self._conv_block_2 = base.SimpleConvBlock(hdim=self.hdim, odim=self.hdim, kernel_size=5,
                                                  **kwargs)
        self._conv_block_3 = base.SimpleConvBlock(hdim=self.hdim, odim=self.hdim, kernel_size=5,
                                                  **kwargs)

    def __call__(self, states):
        return jax.nn.relu(self._conv_block_3(jax.nn.relu(
            self._conv_block_2(jax.nn.relu(self._conv_block_1(self._to_black(states)))))))
