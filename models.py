import haiku as hk
import jax.random


class RandomGoModel(hk.Module):
    def __call__(self, x):
        return jax.random.normal(hk.next_rng_key(), (x.shape[0], x.shape[2] * x.shape[3] + 1))


def get_model(model_class: str) -> hk.Transformed:
    return hk.transform(lambda states: {'random': RandomGoModel}[model_class]()(states))
