"""
Models that map Go states to other vector spaces, which can be used for feature extraction or
dimensionality reduction.
"""

import gojax
import haiku as hk
import jax.nn
import jax.numpy as jnp

from muzero_gojax.models import base


class IdentityEmbed(base.BaseGoModel):
    """IdentityEmbed model. Should be used with the real transition."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.model_params.embed_dim == gojax.NUM_CHANNELS

    def __call__(self, states):
        return states.astype(self.model_params.dtype)


class AmplifiedEmbed(base.BaseGoModel):
    """Amplifies the range of the state from {0, 1} to {-1, 1}."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.model_params.embed_dim == gojax.NUM_CHANNELS

    def __call__(self, states):
        return states.astype(self.model_params.dtype) * 2 - 1


class BlackPerspectiveEmbed(base.BaseGoModel):
    """Converts all states whose turn is white to black's perspective."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.model_params.embed_dim == gojax.NUM_CHANNELS

    def __call__(self, states):
        return jnp.where(jnp.expand_dims(gojax.get_turns(states), (1, 2, 3)),
                         gojax.swap_perspectives(states),
                         states).astype(self.model_params.dtype)


class NonSpatialConvEmbed(base.BaseGoModel):
    """A light-weight CNN neural network."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv = base.NonSpatialConv(hdim=self.model_params.hdim,
                                         odim=self.model_params.embed_dim,
                                         nlayers=0)

    def __call__(self, states):
        return self._conv(states.astype(self.model_params.dtype))


class CnnLiteEmbed(base.BaseGoModel):
    """A light-weight CNN neural network."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._simple_conv_block = base.SimpleConvBlock(
            hdim=self.model_params.hdim,
            odim=self.model_params.embed_dim,
            **kwargs)

    def __call__(self, states):
        return jax.nn.relu(
            self._simple_conv_block(states.astype(self.model_params.dtype)))


class BlackCnnLiteEmbed(base.BaseGoModel):
    """Black perspective embedding followed by a light-weight CNN neural network."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._to_black = BlackPerspectiveEmbed(*args, **kwargs)
        self._simple_conv_block = base.SimpleConvBlock(
            hdim=self.model_params.hdim,
            odim=self.model_params.embed_dim,
            **kwargs)

    def __call__(self, states):
        return jax.nn.relu(
            self._simple_conv_block(
                self._to_black(states).astype(self.model_params.dtype)))


class ResNetV2Embed(base.BaseGoModel):
    """ResNetV2 model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resnet = base.ResNetV2(hdim=self.model_params.hdim,
                                     nlayers=self.model_params.nlayers,
                                     odim=self.model_params.hdim)
        self._conv = hk.Conv2D(self.model_params.embed_dim, (1, 1),
                               data_format='NCHW')

    def __call__(self, embeds):
        return self._conv(self._resnet(embeds.astype(self.model_params.dtype)))
