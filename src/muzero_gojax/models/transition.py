"""Models that map state embeddings to the next state embeddings for all actions."""
import gojax
import haiku as hk
import jax
import jax.numpy as jnp
from jax import nn

from muzero_gojax.models import base
from muzero_gojax.models import embed


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
        states = jnp.reshape(jnp.repeat(jnp.expand_dims(states, 1), action_size, axis=1), (
            batch_size * action_size, gojax.NUM_CHANNELS, board_height, board_width))
        indicator_actions = jnp.reshape(
            nn.one_hot(jnp.repeat(jnp.arange(action_size), batch_size), num_classes=action_size - 1,
                       dtype=bool), (batch_size * action_size, board_height, board_width))
        return jnp.reshape(gojax.next_states(states, indicator_actions),
                           (batch_size, action_size, gojax.NUM_CHANNELS, board_height, board_width))


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
        self._simple_conv_block = base.SimpleConvBlock(hdim=self.hdim,
                                                       odim=self.hdim * self.action_size, **kwargs)

    def __call__(self, embeds):
        return jnp.reshape(self._simple_conv_block(embeds.astype('bfloat16')), (
            len(embeds), self.action_size, self.hdim, self.board_size, self.board_size))


class CNNIntermediateTransition(base.BaseGoModel):
    """
    1-layer CNN model with hidden and output dimension set to 256.

    Intended to be used the BlackCNNLite embedding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv_block_1 = base.SimpleConvBlock(hdim=self.hdim, odim=self.hdim, **kwargs)
        self._conv_block_2 = base.SimpleConvBlock(hdim=self.hdim, odim=self.hdim, **kwargs)
        self._conv_block_3 = base.SimpleConvBlock(hdim=self.hdim, odim=self.hdim * self.action_size,
                                                  **kwargs)

    def __call__(self, embeds):
        return jnp.reshape(self._conv_block_3(jax.nn.relu(
            self._conv_block_2(jax.nn.relu(self._conv_block_1(embeds.astype('bfloat16')))))),
            (len(embeds), self.action_size, self.hdim, self.board_size, self.board_size))
