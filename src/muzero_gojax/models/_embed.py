"""
Models that map Go states to other vector spaces, which can be used for feature extraction or
dimensionality reduction.
"""

import warnings

import haiku as hk
import jax.numpy as jnp

import gojax
from muzero_gojax.models import _base


class IdentityEmbed(_base.BaseGoModel):
    """IdentityEmbed model. Should be used with the real transition."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.model_config.embed_dim == gojax.NUM_CHANNELS

    def __call__(self, states):
        return states.astype(
            hk.mixed_precision.get_policy(hk.Module).param_dtype)


class AmplifiedEmbed(_base.BaseGoModel):
    """Amplifies the range of the state from {0, 1} to {-1, 1}."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.model_config.embed_dim == gojax.NUM_CHANNELS

    def __call__(self, states):
        return states.astype(
            hk.mixed_precision.get_policy(hk.Module).param_dtype) * 2 - 1


class CanonicalEmbed(_base.BaseGoModel):
    """Converts all states whose turn is white to black's perspective."""

    def __call__(self, states):
        return jnp.where(jnp.expand_dims(gojax.get_turns(states), (1, 2, 3)),
                         gojax.swap_perspectives(states), states).astype(
                             hk.mixed_precision.get_policy(
                                 hk.Module).output_dtype)


class LinearConvEmbed(_base.BaseGoModel):
    """Linear convolution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv = hk.Conv2D(self.model_config.embed_dim, (1, 1),
                               data_format='NCHW')

    def __call__(self, states):
        return self._conv(
            states.astype(
                hk.mixed_precision.get_policy(hk.Module).param_dtype))


class NonSpatialConvEmbed(_base.BaseGoModel):
    """A light-weight CNN neural network."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv = _base.NonSpatialConv(hdim=self.model_config.hdim,
                                          odim=self.model_config.embed_dim,
                                          nlayers=0)

    def __call__(self, states):
        return self._conv(
            states.astype(
                hk.mixed_precision.get_policy(hk.Module).param_dtype))


class BroadcastResNetV2Embed(_base.BaseGoModel):
    """[DEPRECATED] ResNetV2 model with a broadcast layer at the end."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            'BroadcastResNetV2Embed is deprecated. Use ResNetV2Embed instead.')
        self._resnet = _base.ResNetV2(
            hdim=self.model_config.hdim,
            nlayers=self.submodel_config.nlayers,
            odim=self.model_config.hdim,
            broadcast_frequency=self.model_config.broadcast_frequency,
            bottleneck_div=self.model_config.bottleneck_div)
        self._conv = hk.Conv2D(self.model_config.embed_dim, (1, 1),
                               data_format='NCHW')

    def __call__(self, states):
        return self._conv(
            self._resnet(
                states.astype(
                    hk.mixed_precision.get_policy(hk.Module).param_dtype)))


class CanonicalBroadcastResNetV2Embed(_base.BaseGoModel):
    """[DEPRECATED] ResNetV2 model with a canonical lens (black perspective)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn('CanonicalBroadcastResNetV2Embed is deprecated. '
                      'Use CanonicalResNetV2Embed instead.')
        self._canonical = CanonicalEmbed(*args, **kwargs)
        self._resnet = _base.ResNetV2(
            hdim=self.model_config.hdim,
            nlayers=self.submodel_config.nlayers,
            odim=self.model_config.hdim,
            broadcast_frequency=self.model_config.broadcast_frequency,
            bottleneck_div=self.model_config.bottleneck_div)
        self._conv = hk.Conv2D(self.model_config.embed_dim, (1, 1),
                               data_format='NCHW')

    def __call__(self, states):
        return self._conv(self._resnet(self._canonical(states)))


class ResNetV2Embed(_base.BaseGoModel):
    """ResNetV2 model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resnet = _base.ResNetV2(
            hdim=self.model_config.hdim,
            nlayers=self.submodel_config.nlayers,
            odim=self.model_config.hdim,
            broadcast_frequency=self.model_config.broadcast_frequency,
            bottleneck_div=self.model_config.bottleneck_div)
        self._conv = hk.Conv2D(self.model_config.embed_dim, (1, 1),
                               data_format='NCHW')

    def __call__(self, states):
        return self._conv(
            self._resnet(
                states.astype(
                    hk.mixed_precision.get_policy(hk.Module).param_dtype)))


class CanonicalResNetV2Embed(_base.BaseGoModel):
    """RezsNetV2 model with a canonical lens (black perspective)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._canonical = CanonicalEmbed(*args, **kwargs)
        self._resnet = _base.ResNetV2(
            hdim=self.model_config.hdim,
            nlayers=self.submodel_config.nlayers,
            odim=self.model_config.hdim,
            broadcast_frequency=self.model_config.broadcast_frequency,
            bottleneck_div=self.model_config.bottleneck_div)
        self._conv = hk.Conv2D(self.model_config.embed_dim, (1, 1),
                               data_format='NCHW')

    def __call__(self, states):
        return self._conv(self._resnet(self._canonical(states)))


class CanonicalResNetV3Embed(_base.BaseGoModel):
    """My simplified version of ResNet V2."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._canonical = CanonicalEmbed(*args, **kwargs)
        self._resnet = _base.ResNetV3(hdim=self.model_config.hdim,
                                      nlayers=self.submodel_config.nlayers,
                                      odim=self.model_config.embed_dim)

    def __call__(self, states: jnp.ndarray) -> jnp.ndarray:
        return self._resnet(self._canonical(states))
