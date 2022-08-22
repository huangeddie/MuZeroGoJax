"""Models that map state embeddings to the next state embeddings for all actions."""
import gojax
import haiku as hk
import jax
import jax.numpy as jnp

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
        embeds = embeds.astype('bfloat16')
        embed_shape = embeds.shape[1:]
        transition_w = hk.get_parameter('transition_w',
                                        shape=(*embed_shape, self.action_size, *embed_shape),
                                        init=hk.initializers.RandomNormal(1. / self.board_size),
                                        dtype=embeds.dtype)
        transition_b = hk.get_parameter('transition_b', shape=(1, *embed_shape),
                                        init=hk.initializers.Constant(0.), dtype=embeds.dtype)

        return jnp.einsum('bchw,chwakxy->bakxy', embeds, transition_w) + transition_b


class RealTransition(base.BaseGoModel):
    """
    Real Go transitions.

    Should be used with the identity embedding.
    """

    def __call__(self, embeds):
        return gojax.get_children(embeds)


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
        stacked_transitions = jax.nn.relu(self._conv_block_3(jax.nn.relu(
            self._conv_block_2(jax.nn.relu(self._conv_block_1(embeds.astype('bfloat16')))))))
        return jnp.reshape(stacked_transitions, (
        len(embeds), self.action_size, self.hdim, self.board_size, self.board_size))
