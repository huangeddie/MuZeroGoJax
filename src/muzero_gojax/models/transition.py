"""Models that map state embeddings to the next state embeddings for all actions."""

import gojax
import haiku as hk
import jax
import jax.nn
import jax.numpy as jnp
from jax import lax

from muzero_gojax import nt_utils
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

    Should be used with the BlackPerspectiveEmbed embedding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._internal_real_transition = RealTransition(*args, **kwargs)
        self._internal_black_perspective_embed = embed.BlackPerspectiveEmbed(*args, **kwargs)

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
        self._conv = hk.Conv2D(self.model_params.embed_dim * self.action_size, (3, 3),
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
        odim = self.model_params.embed_dim * self.action_size
        self._simple_conv_block = base.SimpleConvBlock(hdim=self.model_params.hdim, odim=odim,
                                                       **kwargs)

    def __call__(self, embeds):
        return jnp.reshape(jax.nn.relu(self._simple_conv_block(embeds.astype('bfloat16'))),
                           self.transition_output_shape)


class ResNetV2Transition(base.BaseGoModel):
    """ResNetV2 model."""

    def __init__(self, *args, **kwargs):
        # pylint: disable=duplicate-code
        super().__init__(*args, **kwargs)
        self._resnet = base.ResNetV2(hdim=self.model_params.hdim, nlayers=self.model_params.nlayers,
                                     odim=self.model_params.hdim)
        self._conv = hk.Conv2D(self.model_params.embed_dim * self.action_size, (1, 1),
                               data_format='NCHW')

    def __call__(self, embeds):
        return jnp.reshape(self._conv(self._resnet(embeds.astype('bfloat16'))),
                           self.transition_output_shape)


class ResNetV2ActionEmbedTransition(base.BaseGoModel):
    """ResNetV2 model."""

    def __init__(self, *args, **kwargs):
        # pylint: disable=duplicate-code
        super().__init__(*args, **kwargs)
        self._resnet = base.ResNetV2(hdim=self.model_params.hdim, nlayers=self.model_params.nlayers,
                                     odim=self.model_params.hdim)
        self._conv = hk.Conv2D(self.model_params.embed_dim, (1, 1), data_format='NCHW')

    def __call__(self, embeds: jnp.ndarray) -> jnp.ndarray:
        # Embeds is N x D x B x B
        # A x B x B
        indicator_actions = gojax.action_1d_to_indicator(jnp.arange(self.action_size),
                                                         self.model_params.board_size,
                                                         self.model_params.board_size)
        # N x A x 1 x B x B
        batch_size = len(embeds)
        batch_indicator_actions = jnp.expand_dims(
            jnp.repeat(jnp.expand_dims(indicator_actions, axis=0), repeats=batch_size, axis=0),
            axis=2).astype('bfloat16')
        # N x A x (D+1) x B x B
        duplicated_embeds = jnp.repeat(jnp.expand_dims(embeds.astype('bfloat16'), axis=1),
                                       repeats=self.action_size, axis=1)
        embeds_with_actions = jnp.concatenate((duplicated_embeds, batch_indicator_actions), axis=2)

        return nt_utils.unflatten_first_dim(
            self._conv(self._resnet(nt_utils.flatten_first_two_dims(embeds_with_actions))),
            batch_size, self.action_size)
