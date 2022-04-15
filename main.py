import haiku as hk
import jax.nn
from jax import numpy as jnp
from gojax import go
from gojax import constants


class GoModel(hk.Module):
    def __call__(self, x):
        return jax.random.normal(hk.next_rng_key(), (x.shape[0], x.shape[2], x.shape[3]))


def _call_go_model(states):
    return GoModel()(states)


def train(model_fn, rng_key):
    return model_fn.init(rng_key, go.new_states(7, 1))


def simulate_next_states(go_model, params, rng_key, states):
    raw_action_logits = go_model.apply(params, rng_key, states)
    action_logits = jnp.where(states[:, constants.INVALID_CHANNEL_INDEX],
                              jnp.full_like(raw_action_logits, float('-inf')), raw_action_logits)
    flattened_action_values = jnp.reshape(action_logits, (1, -1))
    action_1d = jax.random.categorical(rng_key, flattened_action_values)
    one_hot_action_1d = jax.nn.one_hot(action_1d, flattened_action_values.shape[-1], dtype=bool)
    indicator_actions = jnp.reshape(one_hot_action_1d, (-1, action_logits.shape[1], action_logits.shape[2]))
    states = jax.jit(go.next_states)(states, indicator_actions)
    return states


def main():
    go_model = hk.transform(_call_go_model)
    rng_key = jax.random.PRNGKey(42)
    parameters = train(go_model, rng_key)

    board_size = 7
    states = go.new_states(board_size, 1)
    step = 0
    while not jnp.alltrue(go.get_ended(states)):
        states = simulate_next_states(go_model, parameters, rng_key, states)
        rng_key, _ = jax.random.split(rng_key)
        print(f'Step {step}')
        print(go.get_pretty_string(states[0]))
        step += 1


if __name__ == '__main__':
    main()
