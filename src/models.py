"""Model architectures."""
import haiku as hk
import jax.numpy as jnp
import jax.random


class RandomGoModel(hk.Module):
    """Random model where each action logit is an independent random normal scalar variable."""

    def __call__(self, states):
        batch_size = len(states)
        hk.reserve_rng_keys(2)
        return jax.random.normal(hk.next_rng_key(),
                                 (batch_size,
                                  states.shape[2] * states.shape[3] + 1)), jax.random.normal(
            hk.next_rng_key(), (batch_size,))


class LinearGoModel(hk.Module):
    """Linear model."""

    def __call__(self, x):
        board_size = x.shape[-1]
        batch_size = len(x)
        x = jnp.reshape(x, (batch_size, -1))
        hdim = x.shape[-1]
        action_w = hk.get_parameter("action_w",
                                    shape=(hdim, board_size ** 2 + 1),
                                    init=hk.initializers.RandomNormal(1. / board_size))
        value_w = hk.get_parameter("value_w",
                                   shape=(hdim,),
                                   init=hk.initializers.RandomNormal(1. / board_size))
        value_b = hk.get_parameter("value_b", shape=(), init=jnp.zeros)
        return jnp.dot(x, action_w), (jnp.dot(x, value_w) + value_b)


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
