"""Model architectures."""
import haiku as hk
import jax.random


class RandomGoModel(hk.Module):
    """Random model where each action logit is an indepedent random normal scalar variable."""

    def __call__(self, states):
        return jax.random.normal(hk.next_rng_key(),
                                 (states.shape[0], states.shape[2] * states.shape[3] + 1))


def get_model(model_class: str) -> hk.Transformed:
    """
    Gets the corresponding model for the given name.
    :param model_class: Name of the model class.
    :return: A Haiku-transformed Go model.
    """
    # pylint: disable=unnecessary-lambda
    return hk.transform(lambda states: {'random': RandomGoModel}[model_class]()(states))
