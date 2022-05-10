"""Models that map state embeddings to state-action policy logits."""

import haiku as hk
import jax
import jax.numpy as jnp

from models import base_go_model


class RandomPolicy(base_go_model.BaseGoModel):
    """Outputs independent standard normal variables."""

    def __call__(self, state_embeds):
        return jax.random.normal(hk.next_rng_key(), (len(state_embeds), self.action_size))


class Linear3DPolicy(base_go_model.BaseGoModel):
    """Linear model."""

    def __call__(self, state_embeds):
        action_w = hk.get_parameter('action_w',
                                    shape=state_embeds.shape[1:] + (self.action_size,),
                                    init=hk.initializers.RandomNormal(1. / self.board_size))

        return jnp.einsum('bchw,chwa->ba', state_embeds, action_w)
