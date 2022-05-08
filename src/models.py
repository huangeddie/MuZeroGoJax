"""Model architectures."""
import gojax
import haiku as hk
import jax.numpy as jnp
import jax.random


class RandomGoModel(hk.Module):
    """Random model where each action logit is an independent random normal scalar variable."""

    def __call__(self, states):
        batch_size = len(states)
        action_size = states.shape[2] * states.shape[3] + 1
        hk.reserve_rng_keys(3)
        policy_logits = jax.random.normal(hk.next_rng_key(),
                                          (batch_size, action_size))
        value_logits = jax.random.normal(
            hk.next_rng_key(), (batch_size,))
        transition_logits = jax.random.normal(hk.next_rng_key(), (
            batch_size, action_size, gojax.NUM_CHANNELS, states.shape[2], states.shape[3]))
        return policy_logits, value_logits, transition_logits


class LinearGoModel(hk.Module):
    """Linear model."""

    def __call__(self, states):
        board_size = states.shape[-1]
        action_size = states.shape[-2] * states.shape[-1] + 1
        action_w = hk.get_parameter('action_w',
                                    shape=states.shape[1:] + (action_size,),
                                    init=hk.initializers.RandomNormal(1. / board_size))
        value_w = hk.get_parameter('value_w', shape=states.shape[1:],
                                   init=hk.initializers.RandomNormal(1. / board_size))
        value_b = hk.get_parameter('value_b', shape=(), init=jnp.zeros)
        transition_w = hk.get_parameter('transition_w',
                                        shape=states.shape[1:] + (action_size,) + states.shape[1:],
                                        init=hk.initializers.RandomNormal(1. / board_size))
        transition_b = hk.get_parameter('transition_b', shape=states.shape[1:], init=jnp.zeros)
        policy_logits = jnp.einsum('bchw,chwa->ba', states, action_w)
        value_logits = jnp.einsum('bchw,chw->b', states, value_w) + value_b
        transition_logits = jnp.einsum('bchw,chwakxy->akxy', states, transition_w) + transition_b
        return policy_logits, value_logits, transition_logits


def get_model(model_class: str) -> hk.Transformed:
    """
    Gets the corresponding model for the given name.
    :param model_class: Name of the model class.
    :return: A Haiku-transformed Go model.
    """
    # pylint: disable=unnecessary-lambda
    model_dict = {'random': RandomGoModel, 'linear': LinearGoModel}
    return hk.transform(
        lambda states: model_dict[model_class]()(states))
