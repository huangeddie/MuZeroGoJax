"""Models that map state embeddings to the next state embeddings for all actions."""

import gojax
import haiku as hk
import jax
import jax.numpy as jnp
from jax import lax

from muzero_gojax.models import base
from muzero_gojax.models import embed


class RandomTransition(base.BaseGoModel):
    """Outputs independent standard normal variables."""

    def __call__(self, embeds):
        return jax.random.normal(hk.next_rng_key(),
                                 (len(embeds), *self.transition_output_shape[1:]), dtype='bfloat16')


class RealTransition(base.BaseGoModel):
    """
    Real Go transitions.

    Should be used with the identity embedding.
    """

    def __call__(self, embeds):
        return lax.stop_gradient(gojax.get_children(embeds.astype(bool)).astype('bfloat16'))


class BlackRealTransition(base.BaseGoModel):
    """
    Real Go transitions under black's perspective.

    Should be used with the BlackPerspective embedding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._internal_real_transition = RealTransition(*args, **kwargs)
        self._internal_black_perspective_embed = embed.BlackPerspective(*args, **kwargs)

    def __call__(self, embeds):
        transitions = self._internal_real_transition(embeds)
        batch_size, action_size, channel, board_height, board_width = transitions.shape
        black_perspectives = self._internal_black_perspective_embed(
            jnp.reshape(transitions.astype(bool),
                        (batch_size * action_size, channel, board_height, board_width)))
        return lax.stop_gradient(jnp.reshape(black_perspectives, transitions.shape))


class LinearConvTransition(base.BaseGoModel):
    """Linear model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv = hk.Conv2D(self.absl_flags.embed_dim * self.action_size, (3, 3),
                               data_format='NCHW')

    def __call__(self, embeds):
        return jnp.reshape(self._conv(embeds.astype('bfloat16')), self.transition_output_shape)


class CnnLiteTransition(base.BaseGoModel):
    """
    1-layer CNN model with hidden and output dimension.

    Intended to be used the BlackCNNLite embedding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._simple_conv_block = base.SimpleConvBlock(hdim=self.absl_flags.hdim,
                                                       odim=self.absl_flags.embed_dim * self.action_size,
                                                       **kwargs)

    def __call__(self, embeds):
        return jnp.reshape(jax.nn.relu(self._simple_conv_block(embeds.astype('bfloat16'))),
                           self.transition_output_shape)


class CnnIntermediateTransition(base.BaseGoModel):
    """
    3-layer CNN model with hidden and output dimension.

    Intended to be used the BlackCNNLite embedding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv_block_1 = base.SimpleConvBlock(hdim=self.absl_flags.hdim,
                                                  odim=self.absl_flags.hdim, **kwargs)
        self._conv_block_2 = base.SimpleConvBlock(hdim=self.absl_flags.hdim,
                                                  odim=self.absl_flags.hdim, **kwargs)
        self._conv_block_3 = base.SimpleConvBlock(hdim=self.absl_flags.hdim,
                                                  odim=self.absl_flags.embed_dim * self.action_size,
                                                  **kwargs)

    def __call__(self, embeds):
        stacked_transitions = jax.nn.relu(self._conv_block_3(jax.nn.relu(
            self._conv_block_2(jax.nn.relu(self._conv_block_1(embeds.astype('bfloat16')))))))
        return jnp.reshape(stacked_transitions, self.transition_output_shape)


class ResnetIntermediateTransition(base.BaseGoModel):
    """
    3-layer CNN model with hidden and output dimension.

    Intended to be used the BlackCNNLite embedding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initial_conv = hk.Conv2D(self.absl_flags.hdim, (3, 3), data_format='NCHW')
        self.blocks = [base.ResBlockV2(channels=self.absl_flags.hdim, **kwargs),
                       base.ResBlockV2(channels=self.absl_flags.hdim, **kwargs),
                       base.ResBlockV2(channels=self.absl_flags.embed_dim * self.action_size,
                                       use_projection=True, **kwargs), ]
        self._final_layer_norm = hk.LayerNorm(axis=(1, 2, 3), create_scale=False,
                                              create_offset=False)

    def __call__(self, embeds):
        out = self._initial_conv(embeds.astype('bfloat16'))
        for block in self.blocks:
            out = jax.nn.relu(block(out))
        out = jax.nn.relu(self._final_layer_norm(out))
        return jnp.reshape(out, self.transition_output_shape)
