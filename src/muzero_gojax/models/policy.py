"""Models that map state embeddings to state-action policy logits."""

import gojax
import haiku as hk
import jax
import jax.numpy as jnp

from muzero_gojax.models import base
from muzero_gojax.models import value


class RandomPolicy(base.BaseGoModel):
    """Outputs independent standard normal variables."""

    def __call__(self, embeds):
        return jax.random.normal(hk.next_rng_key(), (len(embeds), self.action_size))


class Linear3DPolicy(base.BaseGoModel):
    """Linear model."""

    def __call__(self, embeds):
        embeds = embeds.astype('bfloat16')
        action_w = hk.get_parameter('action_w', shape=embeds.shape[1:] + (self.action_size,),
                                    init=hk.initializers.RandomNormal(1. / self.board_size))

        return jnp.einsum('bchw,chwa->ba', embeds, action_w)


class CNNLitePolicy(base.BaseGoModel):
    """
    Single layer 1x1 CNN network with 32 hidden dimensions.

    Assumes an N x C x H x W input.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._simple_conv_block = base.SimpleConvBlock(hdim=self.hdim, odim=1, use_batch_norm=False, **kwargs)
        self._pass_value = value.Linear3DValue(*args, **kwargs)

    def __call__(self, embeds):
        float_embeds = embeds.astype('bfloat16')
        move_logits = self._simple_conv_block(float_embeds)
        pass_logits = self._pass_value(float_embeds)
        return jnp.concatenate(
            (jnp.reshape(move_logits, (len(embeds), self.action_size - 1)), jnp.expand_dims(pass_logits, 1)), axis=1)


class TrompTaylorPolicy(base.BaseGoModel):
    """
    Logits equal to player's area - opponent's area for next state.

    Requires identity embedding.
    """

    def __call__(self, embeds):
        all_children = gojax.get_children(embeds)
        batch_size, action_size, channels, nrows, ncols = all_children.shape
        turns = jnp.repeat(jnp.expand_dims(gojax.get_turns(embeds), axis=1), repeats=action_size, axis=1)
        flat_children = jnp.reshape(all_children, (batch_size * action_size, channels, nrows, ncols))
        flat_turns = jnp.reshape(turns, batch_size * action_size)
        sizes = gojax.compute_area_sizes(flat_children).astype('bfloat16')
        n_idcs = jnp.arange(len(sizes))
        return jnp.reshape(sizes[n_idcs, flat_turns.astype('uint8')] - sizes[n_idcs, (~flat_turns).astype('uint8')],
                           (batch_size, action_size))
