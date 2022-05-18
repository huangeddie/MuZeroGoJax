"""Models that map state embeddings to state value logits."""

import haiku as hk
import jax
import jax.numpy as jnp

from models import base


class RandomValue(base.BaseGoModel):
    """Outputs independent standard normal variables."""

    def __call__(self, state_embeds):
        return jax.random.normal(hk.next_rng_key(), (len(state_embeds),))


class Linear3DValue(base.BaseGoModel):
    """Linear model."""

    def __call__(self, state_embeds):
        value_w = hk.get_parameter('value_w',
                                   shape=state_embeds.shape[1:],
                                   init=hk.initializers.RandomNormal(1. / self.board_size))
        value_b = hk.get_parameter('value_b', shape=(), init=jnp.zeros)

        return jnp.einsum('bchw,chw->b', state_embeds, value_w) + value_b
