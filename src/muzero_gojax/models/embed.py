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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.absl_flags.embed_dim == gojax.NUM_CHANNELS

    def __call__(self, states):
        return states.astype('bfloat16')


class BlackPerspective(base.BaseGoModel):
    """Converts all states whose turn is white to black's perspective."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.absl_flags.embed_dim == gojax.NUM_CHANNELS

    def __call__(self, states):
        return jnp.where(jnp.expand_dims(gojax.get_turns(states), (1, 2, 3)),
                         gojax.swap_perspectives(states), states).astype('bfloat16')


class LinearConvEmbed(base.BaseGoModel):
    """A light-weight CNN neural network."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv = hk.Conv2D(self.absl_flags.embed_dim, (3, 3), data_format='NCHW')

    def __call__(self, states):
        return self._conv(states.astype('bfloat16'))


class CnnLiteEmbed(base.BaseGoModel):
    """A light-weight CNN neural network."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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


class ResNetV2Embed(base.BaseGoModel):
    """ResNetV2 model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resnet_medium = base.ResNetV2(hdim=self.absl_flags.hdim,
                                            nlayers=self.absl_flags.nlayers,
                                            odim=self.absl_flags.hdim)
        self._conv = hk.Conv2D(self.absl_flags.embed_dim, (1, 1), data_format='NCHW')

    def __call__(self, embeds):
        return self._conv(self._resnet_medium(embeds.astype('bfloat16')))
