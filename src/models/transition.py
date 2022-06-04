"""Models that map state embeddings to the next state embeddings for all actions."""
import gojax
import haiku as hk
import jax
import jax.numpy as jnp
from jax import nn

import models.embed
from models import base


class RandomTransition(base.BaseGoModel):
    """Outputs independent standard normal variables."""

    def __call__(self, embeds):
        return jax.random.normal(hk.next_rng_key(),
                                 (len(embeds), self.action_size) + embeds.shape[1:])


class Linear3DTransition(base.BaseGoModel):
    """Linear model."""

    def __call__(self, embeds):
        embed_shape = embeds.shape[1:]
        transition_w = hk.get_parameter('transition_w',
                                        shape=embed_shape + (self.action_size,) + embed_shape,
                                        init=hk.initializers.RandomNormal(1. / self.board_size))
        transition_b = hk.get_parameter('transition_b', shape=embed_shape, init=jnp.zeros)

        return jnp.einsum('bchw,chwakxy->bakxy', embeds, transition_w) + transition_b


class RealTransition(base.BaseGoModel):
    """
    Real Go transitions.

    Should be used with the identity embedding.
    """

    def __call__(self, embeds):
        states = embeds
        batch_size = len(states)
        board_height, board_width = states.shape[2:4]
        action_size = board_height * board_width + 1
        states = jnp.reshape(
            jnp.repeat(jnp.expand_dims(states, 1), action_size, axis=1),
            (batch_size * action_size, gojax.NUM_CHANNELS, board_height,
             board_width))
        indicator_actions = jnp.reshape(
            nn.one_hot(jnp.repeat(jnp.arange(action_size), batch_size), num_classes=action_size - 1,
                       dtype=bool),
            (batch_size * action_size, board_height, board_width))
        return jnp.reshape(gojax.next_states(states, indicator_actions), (
            batch_size, action_size, gojax.NUM_CHANNELS, board_height, board_width))


class BlackPerspectiveRealTransition(base.BaseGoModel):
    """
    Real Go transitions under black's perspective.

    Should be used with the BlackPerspective embedding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._internal_real_transition = RealTransition(*args, **kwargs)
        self._internal_black_perspective_embed = models.embed.BlackPerspective(*args, **kwargs)

    def __call__(self, embeds):
        transitions = self._internal_real_transition(embeds)
        batch_size, action_size, channel, board_height, board_width = transitions.shape
        black_perspectives = self._internal_black_perspective_embed(jnp.reshape(transitions, (
            batch_size * action_size, channel, board_height, board_width)))
        return jnp.reshape(black_perspectives, transitions.shape)


class CNNLiteTransition(base.BaseGoModel):
    """
    1-layer CNN model with hidden and output dimension set to 32.

    Intended to be used the BlackCNNLite embedding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv1 = hk.Conv2D(32, (3, 3), data_format='NCHW')
        self._conv2 = hk.Conv2D(32 * self.action_size, (3, 3), data_format='NCHW')

    def __call__(self, embeds):
        return jnp.reshape(self._conv2(jax.nn.relu(self._conv1(embeds))),
                           (len(embeds), self.action_size, 32, self.board_size, self.board_size))
