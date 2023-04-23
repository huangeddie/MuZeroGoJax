"""Models that map state embeddings to state value logits."""

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

import gojax
from muzero_gojax.models import _base


class RandomValue(_base.BaseGoModel):
    """Outputs independent standard normal variables."""

    def __call__(self, embeds):
        return jax.random.normal(hk.next_rng_key(),
                                 (len(embeds), 2, self.model_config.board_size,
                                  self.model_config.board_size))


class NonSpatialConvValue(_base.BaseGoModel):
    """Non-spatial convolution model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv = _base.NonSpatialConv(hdim=self.model_config.hdim,
                                          odim=2,
                                          nlayers=self.submodel_config.nlayers)

    def __call__(self, embeds):
        return self._conv(embeds.astype(self.model_config.dtype))


class LinearConvValue(_base.BaseGoModel):
    """Linear convolution model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv = hk.Conv2D(2, (1, 1), data_format='NCHW')

    def __call__(self, embeds):
        return self._conv(embeds.astype(self.model_config.dtype))


class SingleLayerConvValue(_base.BaseGoModel):
    """LayerNorm -> ReLU -> Conv."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._layer_norm = hk.LayerNorm(axis=(1, 2, 3),
                                        create_scale=True,
                                        create_offset=True)
        self._conv = _base.NonSpatialConv(hdim=self.model_config.hdim,
                                          odim=2,
                                          nlayers=1)

    def __call__(self, embeds):
        out = embeds.astype(self.model_config.dtype)
        out = self._layer_norm(out)
        out = jax.nn.relu(out)
        return self._conv(out)


class Linear3DValue(_base.BaseGoModel):
    """Linear model."""

    def __call__(self, embeds):
        embeds = embeds.astype(self.model_config.dtype)
        value_w = hk.get_parameter(
            'value_w',
            shape=(*embeds.shape[1:], 2, self.model_config.board_size,
                   self.model_config.board_size),
            init=hk.initializers.RandomNormal(
                1. / self.model_config.board_size / np.sqrt(embeds.shape[1])),
            dtype=embeds.dtype)
        value_b = hk.get_parameter('value_b',
                                   shape=(1, 2, self.model_config.board_size,
                                          self.model_config.board_size),
                                   init=hk.initializers.Constant(0.),
                                   dtype=embeds.dtype)

        return jnp.einsum('bchw,chwxyz->bxyz', embeds, value_w) + value_b


class ResNetV2Value(_base.BaseGoModel):
    """ResNetV2 model."""

    def __init__(self, *args, **kwargs):
        # pylint: disable=duplicate-code
        super().__init__(*args, **kwargs)
        self._resnet = _base.ResNetV2(
            hdim=self.model_config.hdim,
            nlayers=self.submodel_config.nlayers,
            odim=self.model_config.hdim,
            bottleneck_div=self.model_config.bottleneck_div)
        self._non_spatial_conv = hk.Conv2D(2, (1, 1), data_format='NCHW')

    def __call__(self, embeds):
        return self._non_spatial_conv(
            self._resnet(embeds.astype(self.model_config.dtype)))


class TrompTaylorValue(_base.BaseGoModel):
    """
    Player's area - opponent's area.

    Requires identity embedding.
    """

    def __call__(self, embeds):
        states = embeds.astype(bool)
        areas = gojax.compute_areas(states).astype(self.model_config.dtype)
        my_areas = jnp.where(
            jnp.expand_dims(gojax.get_turns(states), (1, 2, 3)),
            areas[:, [1, 0]], areas)
        return (my_areas * 2 - 1) * 100


class ResNetV3Value(_base.BaseGoModel):
    """My simplified version of ResNet V2."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._blocks = [
            _base.ResNetBlockV3(output_channels=self.model_config.embed_dim,
                                hidden_channels=self.model_config.hdim),
            _base.ResNetBlockV3(output_channels=self.model_config.embed_dim,
                                hidden_channels=self.model_config.hdim),
            hk.Conv2D(2, (1, 1), data_format='NCHW'),
        ]

    def __call__(self, embeds: jnp.ndarray) -> jnp.ndarray:
        out = embeds
        for block in self._blocks:
            out = block(out)
        return out
