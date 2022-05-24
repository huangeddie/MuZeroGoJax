"""Models that map state embeddings to state-action policy logits."""

import haiku as hk
import jax
import jax.numpy as jnp

from models import base


class RandomPolicy(base.BaseGoModel):
    """Outputs independent standard normal variables."""

    def __call__(self, embeds):
        return jax.random.normal(hk.next_rng_key(), (len(embeds), self.action_size))


class Linear3DPolicy(base.BaseGoModel):
    """Linear model."""

    def __call__(self, embeds):
        action_w = hk.get_parameter('action_w',
                                    shape=embeds.shape[1:] + (self.action_size,),
                                    init=hk.initializers.RandomNormal(1. / self.board_size))

        return jnp.einsum('bchw,chwa->ba', embeds, action_w)
