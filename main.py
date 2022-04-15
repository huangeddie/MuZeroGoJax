import haiku as hk
import jax.nn
from jax import numpy as jnp
from gojax import go
from gojax import constants


class GoModel(hk.Module):
    def __call__(self, x):
        rng_key = hk.next_rng_key()
        return jax.random.normal(rng_key, (x.shape[0], x.shape[2], x.shape[3]))


def _call_go_model(states):
    return GoModel()(states)


def train(model_fn, rng_key):
    return model_fn.init(rng_key, go.new_states(7, 1))


if __name__ == '__main__':
    call_go_model = hk.transform(_call_go_model)
    rng_key = jax.random.PRNGKey(42)
    parameters = train(call_go_model, rng_key)

    board_size = 7
    states = go.new_states(board_size, 1)
    while not states[0, constants.END_CHANNEL_INDEX, 0, 0]:
        rng_key, sub_key = jax.random.split(rng_key)
        action_exp_logits = jnp.exp(jax.jit(call_go_model.apply)(parameters, sub_key, states)) * jnp.array(
            ~states[:, constants.INVALID_CHANNEL_INDEX], dtype=float)
        flattened_action_values = jnp.reshape(action_exp_logits, (1, -1))
        action_1d = jnp.argmax(flattened_action_values, axis=-1)
        one_hot_action_1d = jax.nn.one_hot(action_1d, board_size ** 2, dtype=bool)
        indicator_actions = jnp.reshape(one_hot_action_1d, (-1, board_size, board_size))
        states = jax.jit(go.next_states)(states, indicator_actions)
        print(go.get_pretty_string(states[0]))
