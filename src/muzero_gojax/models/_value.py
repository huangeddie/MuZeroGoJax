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
        return jax.random.normal(
            hk.next_rng_key(),
            (len(embeds), 2, embeds.shape[-1], embeds.shape[-1]))


class LinearConvValue(_base.BaseGoModel):
    """Linear convolution model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv = hk.Conv2D(2, (1, 1), data_format='NCHW')

    def __call__(self, embeds):
        return self._conv(embeds)


class Linear3DValue(_base.BaseGoModel):
    """Linear model."""

    def __call__(self, embeds):
        value_w = hk.get_parameter(
            'value_w',
            shape=(*embeds.shape[1:], 2, self.model_config.board_size,
                   self.model_config.board_size),
            init=hk.initializers.RandomNormal(
                1. / self.model_config.board_size / np.sqrt(embeds.shape[1])))
        value_b = hk.get_parameter('value_b',
                                   shape=(1, 2, self.model_config.board_size,
                                          self.model_config.board_size),
                                   init=hk.initializers.Constant(0.))

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
        return self._non_spatial_conv(self._resnet(embeds))


class TrompTaylorValue(_base.BaseGoModel):
    """
    Player's area - opponent's area.

    Requires identity embedding.
    """

    def __call__(self, embeds):
        states = embeds.astype(bool)
        areas = gojax.compute_areas(states)
        my_areas = jnp.where(
            jnp.expand_dims(gojax.get_turns(states), (1, 2, 3)),
            areas[:, [1, 0]], areas)
        return ((my_areas * 2 - 1) * 100).astype(jnp.float32)


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
