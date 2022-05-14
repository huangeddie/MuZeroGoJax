"""Models that map state embeddings to the next state embeddings for all actions."""
import gojax
import haiku as hk
import jax
import jax.numpy as jnp
from jax import nn

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


class RealTransition(base_go_model.BaseGoModel):
    """Real Go transitions. Should be used with the identity embedding."""

    def __call__(self, state_embeds):
        states = state_embeds
        batch_size = len(states)
        board_height = states.shape[2]
        board_width = states.shape[3]
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
