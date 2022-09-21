"""Models that try to map Go embeddings back to their states."""

import gojax
import haiku as hk
import jax.numpy as jnp

from muzero_gojax.models import base


class NoOpDecode(base.BaseGoModel):
    """NoOp decoder."""

    def __call__(self, embeds: jnp.ndarray) -> jnp.ndarray:
        """
        Returns an empty states no matter what.

        :param embeds: N x C x B x B float array.
        :return:
        """
        return gojax.new_states(board_size=embeds.shape[2], batch_size=len(embeds)).astype(
            'bfloat16')


class ResNetV2Decode(base.BaseGoModel):
    """ResNetV2 model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resnet_medium = base.ResNetV2(hdim=self.absl_flags.hdim,
                                            nlayers=self.absl_flags.nlayers,
                                            odim=self.absl_flags.hdim)
        self._conv = hk.Conv2D(gojax.NUM_CHANNELS, (1, 1), data_format='NCHW')

    def __call__(self, embeds: jnp.ndarray) -> jnp.ndarray:
        return self._conv(self._resnet_medium(embeds.astype('bfloat16')))
