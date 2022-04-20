import haiku as hk
import jax.nn
from jax import numpy as jnp
from gojax import go


class GoModel(hk.Module):
    def __call__(self, x):
        return jax.random.normal(hk.next_rng_key(), (x.shape[0], x.shape[2], x.shape[3]))


def simulate_next_states(model_fn, params, rng_key, states):
    raw_action_logits = model_fn.apply(params, rng_key, states)
    action_logits = jnp.where(go.get_invalids(states),
                              jnp.full_like(raw_action_logits, float('-inf')), raw_action_logits)
    flattened_action_values = jnp.reshape(action_logits, (states.shape[0], -1))
    action_1d = jax.random.categorical(rng_key, flattened_action_values)
    one_hot_action_1d = jax.nn.one_hot(action_1d, flattened_action_values.shape[-1], dtype=bool)
    indicator_actions = jnp.reshape(one_hot_action_1d, (-1, action_logits.shape[1], action_logits.shape[2]))
    states = jax.jit(go.next_states)(states, indicator_actions)
    return states


def self_play(model_fn, params, batch_size, board_size, rng_key):
    states = go.new_states(board_size, batch_size)
    step = 0
    max_num_steps = 2 * (board_size ** 2)
    trajectories = jnp.repeat(jnp.expand_dims(states, axis=1), max_num_steps, 1).at[:, step].set(states)
    while not jnp.alltrue(go.get_ended(states)) and step <= max_num_steps:
        states = simulate_next_states(model_fn, params, rng_key, states)
        rng_key, _ = jax.random.split(rng_key)
        step += 1
        trajectories = trajectories.at[:, step].set(states)
    return trajectories


def get_winners(trajectories):
    raise NotImplementedError()


def update_params(params, trajectories):
    num_steps = trajectories.shape[1]
    odd_steps = jnp.arange(num_steps // 2) * 2 + 1
    white_perspective_negation = jnp.ones((len(trajectories), num_steps)).at[:, odd_steps].set(-1)
    state_labels = white_perspective_negation * get_winners(trajectories)
    NotImplementedError()
    return params


def train(model_fn, batch_size, board_size, epochs, rng_key):
    params = model_fn.init(rng_key, go.new_states(board_size, 1))

    for _ in range(epochs):
        trajectories = self_play(model_fn, params, batch_size, board_size, rng_key)
        params = update_params(params, trajectories)

    return params


def main():
    go_model = hk.transform(lambda states: GoModel()(states))

    batch_size = 1
    board_size = 7
    epochs = 1
    rng_key = jax.random.PRNGKey(42)
    parameters = train(go_model, batch_size, board_size, epochs, rng_key)

    single_batch_size = 1
    trajectories = self_play(go_model, parameters, single_batch_size, board_size, rng_key)

    for step in range(trajectories.shape[1]):
        print(f'Step {step}')
        print(go.get_pretty_string(trajectories[0, step]))


if __name__ == '__main__':
    main()
