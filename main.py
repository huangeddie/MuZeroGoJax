import haiku as hk
import jax.nn
from gojax import go
from jax import lax
from jax import numpy as jnp


class GoModel(hk.Module):
    def __call__(self, x):
        return jax.random.normal(hk.next_rng_key(), (x.shape[0], x.shape[2], x.shape[3]))


def simulate_next_states(model_fn, params, rng_key, states):
    """
    Simulates the next states of the Go game played out by the given model.

    :param model_fn: a model function that takes in a batch of Go states and parameters and outputs a batch of action
    probabilities for each state.
    :param params: the model parameters.
    :param rng_key: RNG key used to seed the randomness of the simulation.
    :param states: a batch array of N Go games.
    :return: a batch array of N Go games (an N x C x B x B boolean array).
    """
    raw_action_logits = model_fn.apply(params, rng_key, states)
    action_logits = jnp.where(go.get_invalids(states),
                              jnp.full_like(raw_action_logits, float('-inf')), raw_action_logits)
    flattened_action_values = jnp.reshape(action_logits, (states.shape[0], -1))
    action_1d = jax.random.categorical(rng_key, flattened_action_values)
    one_hot_action_1d = jax.nn.one_hot(action_1d, flattened_action_values.shape[-1], dtype=bool)
    indicator_actions = jnp.reshape(one_hot_action_1d, (-1, action_logits.shape[1], action_logits.shape[2]))
    states = jax.jit(go.next_states)(states, indicator_actions)
    return states


def new_trajectories(board_size, batch_size, max_num_steps):
    """
    Creates an empty array of Go game trajectories.

    :param board_size: B.
    :param batch_size: N.
    :param max_num_steps: T.
    :return: an N x T x C x B x B boolean array, where the third dimension (C) contains information about the Go game
    state.
    """
    return jnp.repeat(jnp.expand_dims(go.new_states(board_size, batch_size), axis=1), max_num_steps, 1)


def update_trajectories(model_fn, params, rng_key, step, trajectories):
    """
    Updates the trajectory array for time step `step + 1`.

    :param model_fn: a model function that takes in a batch of Go states and parameters and outputs a batch of action
    probabilities for each state.
    :param params: the model parameters.
    :param rng_key: RNG key which is salted by the time step.
    :param step: the current time step of the trajectory.
    :param trajectories: an N x T x C x B x B boolean array
    :return: an N x T x C x B x B boolean array
    """
    rng_key = jax.random.fold_in(rng_key, step)
    return trajectories.at[:, step + 1].set(
        simulate_next_states(model_fn, params, rng_key, trajectories[:, step]))


def self_play(model_fn, params, batch_size, board_size, rng_key):
    """
    Simulates a batch of trajectories made from playing the model against itself.

    :param model_fn: a model function that takes in a batch of Go states and parameters and outputs a batch of action
    probabilities for each state.
    :param params: the model parameters.
    :param batch_size: N.
    :param board_size: B.
    :param rng_key: RNG key used to seed the randomness of the self play.
    :return: an N x T x C x B x B boolean array.
    """
    max_num_steps = 2 * (board_size ** 2)
    return lax.fori_loop(0, max_num_steps - 1, jax.tree_util.Partial(update_trajectories, model_fn, params, rng_key),
                         new_trajectories(board_size, batch_size, max_num_steps))


def get_winners(trajectories):
    """
    Gets the winner for each trajectory.

    1 = black won
    0 = tie
    -1 = white won

    :param trajectories: an N x T x C x B x B boolean array.
    :return: a boolean array of length N.
    """
    return go.compute_winning(trajectories[:, -1])


def trajectories_to_dataset(trajectories):
    batch_size, num_steps = trajectories.shape[:1]
    state_shape = trajectories.shape[2:]
    odd_steps = jnp.arange(num_steps // 2) * 2 + 1
    white_perspective_negation = jnp.ones((batch_size, num_steps)).at[:, odd_steps].set(-1)
    trajectory_labels = white_perspective_negation * jnp.expand_dims(get_winners(trajectories), 1)
    num_examples = batch_size * num_steps
    states = jnp.reshape(trajectories, (num_examples,) + state_shape)
    state_labels = jnp.reshape(trajectory_labels, (num_examples,))
    return states, state_labels


def update_params(params, trajectories):
    states, state_labels = trajectories_to_dataset(trajectories)
    return params


def train(model_fn, batch_size, board_size, epochs, rng_key):
    params = model_fn.init(rng_key, go.new_states(board_size, 1))

    for _ in range(epochs):
        trajectories = self_play(model_fn, params, batch_size, board_size, rng_key)
        params = update_params(params, trajectories)

    return params


def main():
    go_model = hk.transform(lambda states: GoModel()(states))

    board_size = 3
    batch_size = 1
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
