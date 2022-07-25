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
        return jax.random.normal(hk.next_rng_key(), (len(embeds), self.action_size, *embeds.shape[1:]))


class Linear3DTransition(base.BaseGoModel):
    """Linear model."""

    def __call__(self, embeds):
        embeds = embeds.astype('bfloat16')
        batch_size, hdim, nrows, ncols = embeds.shape
        transition_w = hk.get_parameter('transition_w', shape=(hdim, nrows, ncols, hdim - 1, nrows, ncols),
                                        init=hk.initializers.RandomNormal(1. / self.board_size), dtype=embeds.dtype)
        transition_b = hk.get_parameter('transition_b', shape=(hdim - 1, nrows, ncols),
                                        init=hk.initializers.Constant(0.), dtype=embeds.dtype)

        return jnp.einsum('bdhw,dhwcxy->bcxy', embeds, transition_w) + jnp.expand_dims(transition_b, axis=0)


class RealTransition(base.BaseGoModel):
    """
    Real Go transitions.

    Should be used with the identity embedding.
    """

    def __call__(self, embeds):
        """
        Real Go simulator.

        :param embeds: N x (C+1) x B x B boolean array. Contains Go state and indicator action concatenated.
        :return: Next states
        """
        states = embeds[:, :-1]
        indicator_actions = embeds[:, -1]
        actions = gojax.action_indicator_to_1d(indicator_actions)
        return gojax.next_states(states.astype(bool), actions)


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
        return self._internal_black_perspective_embed(self._internal_real_transition(embeds))


class CNNLiteTransition(base.BaseGoModel):
    """
    1-layer CNN model with hidden and output dimension set to 32.

    Intended to be used the BlackCNNLite embedding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._simple_conv_block = base.SimpleConvBlock(hdim=self.hdim, odim=self.hdim - 1, **kwargs)

    def __call__(self, embeds):
        return self._simple_conv_block(embeds.astype('bfloat16'))


class CNNIntermediateTransition(base.BaseGoModel):
    """
    1-layer CNN model with hidden and output dimension set to 256.

    Intended to be used the BlackCNNLite embedding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv_block_1 = base.SimpleConvBlock(hdim=self.hdim, odim=self.hdim, **kwargs)
        self._conv_block_2 = base.SimpleConvBlock(hdim=self.hdim, odim=self.hdim, **kwargs)
        self._conv_block_3 = base.SimpleConvBlock(hdim=self.hdim, odim=self.hdim - 1, **kwargs)

    def __call__(self, embeds):
        return jax.nn.relu(self._conv_block_3(
            jax.nn.relu(self._conv_block_2(jax.nn.relu(self._conv_block_1(embeds.astype('bfloat16')))))))
