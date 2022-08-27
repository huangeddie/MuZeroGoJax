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


class LinearConvEmbed(base.BaseGoModel):
    """A light-weight CNN neural network."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv = hk.Conv2D(self.absl_flags.embed_dim, (3, 3), data_format='NCHW')

    def __call__(self, states):
        return self._conv(states.astype('bfloat16'))


class CnnIntermediateEmbed(base.BaseGoModel):
    """Black perspective embedding followed by an intermediate CNN neural network."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key_to_remove in ('hdim', 'board_size'):
            if key_to_remove in kwargs:
                kwargs.pop(key_to_remove)
        self._conv_block_1 = base.SimpleConvBlock(hdim=self.absl_flags.hdim,
                                                  odim=self.absl_flags.hdim, **kwargs)
        self._conv_block_2 = base.SimpleConvBlock(hdim=self.absl_flags.hdim,
                                                  odim=self.absl_flags.hdim, **kwargs)
        self._conv_block_3 = base.SimpleConvBlock(hdim=self.absl_flags.hdim,
                                                  odim=self.absl_flags.hdim, **kwargs)
        self._conv_block_4 = base.SimpleConvBlock(hdim=self.absl_flags.hdim,
                                                  odim=self.absl_flags.hdim, **kwargs)
        self._conv_block_5 = base.SimpleConvBlock(hdim=self.absl_flags.hdim,
                                                  odim=self.absl_flags.hdim, **kwargs)
        self._conv_block_6 = base.SimpleConvBlock(hdim=self.absl_flags.hdim,
                                                  odim=self.absl_flags.hdim, **kwargs)
        self._conv_block_7 = base.SimpleConvBlock(hdim=self.absl_flags.hdim,
                                                  odim=self.absl_flags.embed_dim, **kwargs)

    def __call__(self, states):
        out = states.astype('bfloat16')
        out = jax.nn.relu(self._conv_block_1(out))
        out = jax.nn.relu(self._conv_block_2(out))
        out = jax.nn.relu(self._conv_block_3(out))
        out = jax.nn.relu(self._conv_block_4(out))
        out = jax.nn.relu(self._conv_block_5(out))
        out = jax.nn.relu(self._conv_block_6(out))
        out = jax.nn.relu(self._conv_block_7(out))
        return out


class CnnLiteEmbed(base.BaseGoModel):
    """A light-weight CNN neural network."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key_to_remove in ('hdim', 'board_size'):
            if key_to_remove in kwargs:
                kwargs.pop(key_to_remove)
        self._simple_conv_block = base.SimpleConvBlock(hdim=self.absl_flags.hdim,
                                                       odim=self.absl_flags.embed_dim, **kwargs)

    def __call__(self, states):
        return jax.nn.relu(self._simple_conv_block(states.astype('bfloat16')))


class BlackCnnLite(base.BaseGoModel):
    """Black perspective embedding followed by a light-weight CNN neural network."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._to_black = BlackPerspective(*args, **kwargs)
        self._simple_conv_block = base.SimpleConvBlock(hdim=self.absl_flags.hdim,
                                                       odim=self.absl_flags.embed_dim, **kwargs)

    def __call__(self, states):
        return jax.nn.relu(self._simple_conv_block(self._to_black(states).astype('bfloat16')))


class BlackCnnIntermediate(base.BaseGoModel):
    """Black perspective embedding followed by an intermediate CNN neural network."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._to_black = BlackPerspective(*args, **kwargs)
        self._conv_block_1 = base.SimpleConvBlock(hdim=self.absl_flags.hdim,
                                                  odim=self.absl_flags.hdim, **kwargs)
        self._conv_block_2 = base.SimpleConvBlock(hdim=self.absl_flags.hdim,
                                                  odim=self.absl_flags.hdim, **kwargs)
        self._conv_block_3 = base.SimpleConvBlock(hdim=self.absl_flags.hdim,
                                                  odim=self.absl_flags.embed_dim, **kwargs)

    def __call__(self, states):
        return jax.nn.relu(self._conv_block_3(jax.nn.relu(self._conv_block_2(
            jax.nn.relu(self._conv_block_1(self._to_black(states).astype('bfloat16')))))))
