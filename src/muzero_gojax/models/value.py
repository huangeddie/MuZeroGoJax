"""Models that map state embeddings to state value logits."""

import gojax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from muzero_gojax.models import base


class RandomValue(base.BaseGoModel):
    """Outputs independent standard normal variables."""

    def __call__(self, embeds):
        return jax.random.normal(hk.next_rng_key(), (len(embeds), ))


class NonSpatialConvValue(base.BaseGoModel):
    """Non-spatial convolution model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv = base.NonSpatialConv(hdim=self.model_params.hdim,
                                         odim=1,
                                         nlayers=self.model_params.nlayers)

    def __call__(self, embeds):
        embeds = embeds.astype(self.model_params.dtype)
        return jnp.mean(self._conv(embeds), axis=(1, 2, 3))


class Linear3DValue(base.BaseGoModel):
    """Linear model."""

    def __call__(self, embeds):
        embeds = embeds.astype(self.model_params.dtype)
        value_w = hk.get_parameter(
            'value_w',
            shape=embeds.shape[1:],
            init=hk.initializers.RandomNormal(
                1. / self.model_params.board_size / np.sqrt(embeds.shape[1])),
            dtype=embeds.dtype)
        value_b = hk.get_parameter('value_b',
                                   shape=(),
                                   init=hk.initializers.Constant(0.),
                                   dtype=embeds.dtype)

        return jnp.einsum('bchw,chw->b', embeds, value_w) + value_b


class ResNetV2Value(base.BaseGoModel):
    """ResNetV2 model."""

    def __init__(self, *args, **kwargs):
        # pylint: disable=duplicate-code
        super().__init__(*args, **kwargs)
        self._resnet = base.ResNetV2(hdim=self.model_params.hdim,
                                     nlayers=self.model_params.nlayers,
                                     odim=self.model_params.hdim)
        self._non_spatial_conv = base.NonSpatialConv(
            hdim=self.model_params.hdim, nlayers=0, odim=1)

    def __call__(self, embeds):
        return jnp.mean(self._non_spatial_conv(
            self._resnet(embeds.astype(self.model_params.dtype))),
                        axis=(1, 2, 3))


class PieceCounterValue(base.BaseGoModel):
    """
    Player's pieces - opponent's pieces.

    Requires identity embedding.
    """

    def __call__(self, embeds):
        states = embeds.astype(bool)
        turns = gojax.get_turns(states)
        black_pieces = jnp.sum(states[:, gojax.BLACK_CHANNEL_INDEX],
                               dtype=self.model_params.dtype)
        white_pieces = jnp.sum(states[:, gojax.WHITE_CHANNEL_INDEX],
                               dtype=self.model_params.dtype)
        return ((white_pieces - black_pieces) *
                (turns.astype(self.model_params.dtype) * 2 - 1)).astype(
                    self.model_params.dtype)


class TrompTaylorValue(base.BaseGoModel):
    """
    Player's area - opponent's area.

    Requires identity embedding.
    """

    def __call__(self, embeds):
        states = embeds.astype(bool)
        turns = gojax.get_turns(states)
        sizes = gojax.compute_area_sizes(states).astype(
            self.model_params.dtype)
        n_idcs = jnp.arange(len(states))
        return sizes[n_idcs,
                     turns.astype('uint8')] - sizes[n_idcs,
                                                    (~turns).astype('uint8')]
