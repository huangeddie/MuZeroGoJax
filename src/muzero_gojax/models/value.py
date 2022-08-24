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
        return jax.random.normal(hk.next_rng_key(), (len(embeds),))


class Linear3DValue(base.BaseGoModel):
    """Linear model."""

    def __call__(self, embeds):
        embeds = embeds.astype('bfloat16')
        value_w = hk.get_parameter('value_w', shape=embeds.shape[1:],
                                   init=hk.initializers.RandomNormal(
                                       1. / self.board_size / np.sqrt(embeds.shape[1])),
                                   dtype=embeds.dtype)
        value_b = hk.get_parameter('value_b', shape=(), init=hk.initializers.Constant(0.),
                                   dtype=embeds.dtype)

        return jnp.einsum('bchw,chw->b', embeds, value_w) + value_b


class TrompTaylorValue(base.BaseGoModel):
    """
    Player's area - opponent's area.

    Requires identity embedding.
    """

    def __call__(self, embeds):
        turns = gojax.get_turns(embeds)
        sizes = gojax.compute_area_sizes(embeds).astype('bfloat16')
        n_idcs = jnp.arange(len(embeds))
        return sizes[n_idcs, turns.astype('uint8')] - sizes[n_idcs, (~turns).astype('uint8')]
