"""Model architectures."""
import haiku as hk
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


def get_model(model_class: str) -> hk.Transformed:
    """
    Gets the corresponding model for the given name.
    :param model_class: Name of the model class.
    :return: A Haiku-transformed Go model.
    """
    # pylint: disable=unnecessary-lambda
    return hk.transform(lambda states: {'random': RandomGoModel}[model_class]()(states))
