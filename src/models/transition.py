"""Models that map state embeddings to the next state embeddings for all actions."""

import haiku as hk
import jax
import jax.numpy as jnp

from models import base_go_model


class RandomTransition(base_go_model.BaseGoModel):
    """Outputs independent standard normal variables."""

    def __call__(self, state_embeds):
        return jax.random.normal(hk.next_rng_key(),
                                 (len(state_embeds), self.action_size) + state_embeds.shape[1:])


class Linear3DTransition(base_go_model.BaseGoModel):
    """Linear model."""

    def __call__(self, state_embeds):
        embed_shape = state_embeds.shape[1:]
        transition_w = hk.get_parameter('transition_w',
                                        shape=embed_shape + (self.action_size,) + embed_shape,
                                        init=hk.initializers.RandomNormal(1. / self.board_size))
        transition_b = hk.get_parameter('transition_b', shape=embed_shape, init=jnp.zeros)

        return jnp.einsum('bchw,chwakxy->bakxy', state_embeds, transition_w) + transition_b
