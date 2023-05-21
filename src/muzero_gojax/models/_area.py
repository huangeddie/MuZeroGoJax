"""Models that try to map Go embeddings back to their states."""

import haiku as hk
import jax
import jax.numpy as jnp

from muzero_gojax.models import _base


class RandomArea(_base.BaseGoModel):
    """Random model to predict the canonical areas."""

    def __call__(self, embeds):
        return jax.random.normal(hk.next_rng_key(),
                                 (len(embeds), 2, self.model_config.board_size,
                                  self.model_config.board_size))


class LinearConvArea(_base.BaseGoModel):
    """Linear convolution model to predict the canonical areas."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv = hk.Conv2D(2, (1, 1), data_format='NCHW')

    def __call__(self, embeds):
        return self._conv(embeds)


class ResNetV2Area(_base.BaseGoModel):
    """ResNetV2 model."""

    def __init__(self, *args, **kwargs):
        # pylint: disable=duplicate-code
        super().__init__(*args, **kwargs)
        self._resnet = _base.ResNetV2(
            hdim=self.model_config.hdim,
            nlayers=self.submodel_config.nlayers,
            odim=self.model_config.hdim,
            bottleneck_div=self.model_config.bottleneck_div)
        self._conv = hk.Conv2D(2, (1, 1), data_format='NCHW')

    def __call__(self, embeds: jnp.ndarray) -> jnp.ndarray:
        return self._conv(self._resnet(embeds))
