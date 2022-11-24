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
        self._conv = base.NonSpatialConv(hdim=self.model_config.hdim,
                                         odim=1,
                                         nlayers=self.submodel_config.nlayers)

    def __call__(self, embeds):
        embeds = embeds.astype(self.model_config.dtype)
        return jnp.mean(self._conv(embeds), axis=(1, 2, 3))


class LinearConvValue(base.BaseGoModel):
    """Non-spatial linear convolution model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv = base.NonSpatialConv(hdim=self.model_config.hdim,
                                         odim=1,
                                         nlayers=1)

    def __call__(self, embeds):
        embeds = embeds.astype(self.model_config.dtype)
        return jnp.mean(self._conv(embeds), axis=(1, 2, 3))


class SingleLayerConvValue(base.BaseGoModel):
    """LayerNorm -> ReLU -> Conv."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._layer_norm = hk.LayerNorm(axis=(1, 2, 3),
                                        create_scale=True,
                                        create_offset=True)
        self._conv = base.NonSpatialConv(hdim=self.model_config.hdim,
                                         odim=1,
                                         nlayers=1)

    def __call__(self, embeds):
        out = embeds.astype(self.model_config.dtype)
        out = self._layer_norm(out)
        out = jax.nn.relu(out)
        return jnp.mean(self._conv(out), axis=(1, 2, 3))


class NonSpatialQuadConvValue(base.BaseGoModel):
    """Non-spatial quadratic convolution model.

    Should learn to mimic piece counter.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._linear_1 = base.NonSpatialConv(hdim=self.model_config.hdim,
                                             odim=1,
                                             nlayers=1)
        self._linear_2 = base.NonSpatialConv(hdim=self.model_config.hdim,
                                             odim=1,
                                             nlayers=1)

    def __call__(self, embeds):
        embeds = embeds.astype(self.model_config.dtype)
        return jnp.mean(self._linear_1(embeds) * self._linear_2(embeds),
                        axis=(1, 2, 3))


class HeuristicQuadConvValue(base.BaseGoModel):
    """Non-spatial quadratic convolution model.

    Should learn to mimic piece counter.
    """

    def __call__(self, embeds):
        embeds = embeds.astype(self.model_config.dtype)
        w_1 = jnp.zeros((1, gojax.NUM_CHANNELS, 1, 1),
                        dtype=self.model_config.dtype)
        w_1 = w_1.at[0, gojax.BLACK_CHANNEL_INDEX].set(1)
        w_1 = w_1.at[0, gojax.WHITE_CHANNEL_INDEX].set(-1)

        w_2 = jnp.zeros((1, gojax.NUM_CHANNELS, 1, 1),
                        dtype=self.model_config.dtype)
        w_2 = w_2.at[0, gojax.TURN_CHANNEL_INDEX].set(-2)

        o_1 = jax.lax.conv(embeds, w_1, window_strides=(1, 1), padding='same')
        o_2 = jax.lax.conv(embeds, w_2, window_strides=(1, 1),
                           padding='same') + 1
        return jnp.mean(o_1 * o_2, axis=(1, 2, 3))


class Linear3DValue(base.BaseGoModel):
    """Linear model."""

    def __call__(self, embeds):
        embeds = embeds.astype(self.model_config.dtype)
        value_w = hk.get_parameter(
            'value_w',
            shape=embeds.shape[1:],
            init=hk.initializers.RandomNormal(
                1. / self.model_config.board_size / np.sqrt(embeds.shape[1])),
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
        self._resnet = base.ResNetV2(hdim=self.model_config.hdim,
                                     nlayers=self.submodel_config.nlayers,
                                     odim=self.model_config.hdim)
        self._non_spatial_conv = hk.Conv2D(1, (1, 1), data_format='NCHW')

    def __call__(self, embeds):
        return jnp.mean(self._non_spatial_conv(
            self._resnet(embeds.astype(self.model_config.dtype))),
                        axis=(1, 2, 3))


class PieceCounterValue(base.BaseGoModel):
    """
    Player's pieces - opponent's pieces.

    Requires identity embedding.
    """

    def __call__(self, embeds):
        states = embeds.astype(bool)
        turns = gojax.get_turns(states)
        player_is_black = (turns == gojax.BLACKS_TURN)
        black_pieces = jnp.sum(states[:, gojax.BLACK_CHANNEL_INDEX],
                               axis=(1, 2),
                               dtype=self.model_config.dtype)
        white_pieces = jnp.sum(states[:, gojax.WHITE_CHANNEL_INDEX],
                               axis=(1, 2),
                               dtype=self.model_config.dtype)
        return (
            (black_pieces - white_pieces) *
            (player_is_black.astype(self.model_config.dtype) * 2 - 1)).astype(
                self.model_config.dtype)


class TrompTaylorValue(base.BaseGoModel):
    """
    Player's area - opponent's area.

    Requires identity embedding.
    """

    def __call__(self, embeds):
        states = embeds.astype(bool)
        turns = gojax.get_turns(states)
        sizes = gojax.compute_area_sizes(states).astype(
            self.model_config.dtype)
        n_idcs = jnp.arange(len(states))
        return sizes[n_idcs,
                     turns.astype('uint8')] - sizes[n_idcs,
                                                    (~turns).astype('uint8')]
