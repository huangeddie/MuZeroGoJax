"""Models that try to map Go embeddings back to their states."""

import gojax
import haiku as hk
import jax.numpy as jnp

from muzero_gojax.models import base


class AmplifiedDecode(base.BaseGoModel):
    """Amplifies the logit values."""

    def __call__(self, embeds: jnp.ndarray) -> jnp.ndarray:
        # Return the embeds transformed by (x * 200 - 100) if the embeds have the same shape as Go
        # states, which likely means that the embeds ARE the Go states. If they are the Go states,
        # then the values should be {-100, 100}, and the loss should be nearly 0 and the accuracy
        # perfect.
        if embeds.shape[1:] == (
                gojax.NUM_CHANNELS, self.model_params.board_size, self.model_params.board_size):
            return embeds.astype('bfloat16') * 200 - 100
        # Otherwise return an empty batch of Go states with the proper shape.
        return gojax.new_states(board_size=embeds.shape[2], batch_size=len(embeds)).astype(
            'bfloat16')


class LinearConvDecode(base.BaseGoModel):
    """Linear convolution model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv = hk.Conv2D(gojax.NUM_CHANNELS, (3, 3), data_format='NCHW')

    def __call__(self, embeds):
        embeds = embeds.astype('bfloat16')
        return self._conv(embeds.astype('bfloat16'))


class ResNetV2Decode(base.BaseGoModel):
    """ResNetV2 model."""

    def __init__(self, *args, **kwargs):
        # pylint: disable=duplicate-code
        super().__init__(*args, **kwargs)
        self._resnet = base.ResNetV2(hdim=self.model_params.hdim, nlayers=self.model_params.nlayers,
                                     odim=self.model_params.hdim)
        self._conv = hk.Conv2D(gojax.NUM_CHANNELS, (1, 1), data_format='NCHW')

    def __call__(self, embeds: jnp.ndarray) -> jnp.ndarray:
        return self._conv(self._resnet(embeds.astype('bfloat16')))
