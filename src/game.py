"""Manages the model generation of Go games."""
import gojax
import jax.nn
import jax.random
import jax.tree_util
from jax import numpy as jnp, lax


def sample_next_states(model_fn, params, rng_key, states):
    """
    Simulates the next states of the Go game played out by the given model.

    :param model_fn: a model function that takes in a batch of Go states and parameters and
    outputs a batch of action
    probabilities for each state.
    :param params: the model parameters.
    :param rng_key: RNG key used to seed the randomness of the simulation.
    :param states: a batch array of N Go games.
    :return: a batch array of N Go games (an N x C x B x B boolean array).
    """
    raw_action_logits, _, _ = model_fn.apply(params, rng_key, states)
    flattened_invalids = jnp.reshape(gojax.get_invalids(states),
                                     (-1, states.shape[2] * states.shape[3]))
    action_logits = jnp.where(
        jnp.append(flattened_invalids, jnp.zeros((len(states), 1), dtype=bool), axis=1),
        jnp.full_like(raw_action_logits, float('-inf')), raw_action_logits)
    action_1d = jax.random.categorical(rng_key, action_logits)
    one_hot_action_1d = jax.nn.one_hot(action_1d, action_logits.shape[1], dtype=bool)
    indicator_actions = jnp.reshape(one_hot_action_1d[:, :-1],
                                    (-1, states.shape[2], states.shape[3]))
    states = gojax.next_states(states, indicator_actions)
    return states


def new_trajectories(board_size, batch_size, max_num_steps):
    """
    Creates an empty array of Go game trajectories.

    :param board_size: B.
    :param batch_size: N.
    :param max_num_steps: T.
    :return: an N x T x C x B x B boolean array, where the third dimension (C) contains
    information about the Go game
    state.
    """
    return jnp.repeat(jnp.expand_dims(gojax.new_states(board_size, batch_size), axis=1),
                      max_num_steps, 1)


def update_trajectories(model_fn, params, rng_key, step, trajectories):
    """
    Updates the trajectory array for time step `step + 1`.

    :param model_fn: a model function that takes in a batch of Go states and parameters and
    outputs a batch of action
    probabilities for each state.
    :param params: the model parameters.
    :param rng_key: RNG key which is salted by the time step.
    :param step: the current time step of the trajectory.
    :param trajectories: an N x T x C x B x B boolean array
    :return: an N x T x C x B x B boolean array
    """
    rng_key = jax.random.fold_in(rng_key, step)
    return trajectories.at[:, step + 1].set(
        sample_next_states(model_fn, params, rng_key, trajectories[:, step]))


def self_play(model_fn, params, batch_size, board_size, max_num_steps, rng_key):
    # pylint: disable=too-many-arguments
    """
    Simulates a batch of trajectories made from playing the model against itself.

    :param model_fn: a model function that takes in a batch of Go states and parameters and
    outputs a batch of action
    probabilities for each state.
    :param params: the model parameters.
    :param batch_size: N.
    :param board_size: B.
    :param max_num_steps: maximum number of steps.
    :param rng_key: RNG key used to seed the randomness of the self play.
    :return: an N x T x C x B x B boolean array.
    """
    return lax.fori_loop(0, max_num_steps - 1,
                         jax.tree_util.Partial(update_trajectories, model_fn, params, rng_key),
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
    return gojax.compute_winning(trajectories[:, -1])


def trajectories_to_dataset(trajectories):
    """
    Converts trajectories into a dataset.

    The label ({-1, 0, 1}) for the corresponding state represents the winner of the outcome of
    that state's trajectory.

    :param trajectories: An N x T x C x B x B boolean array.
    :return: A batch array of N Go games and an integer array of length N.
    """
    batch_size, num_steps = trajectories.shape[:2]
    state_shape = trajectories.shape[2:]
    odd_steps = jnp.arange(num_steps // 2) * 2 + 1
    white_perspective_negation = jnp.ones((batch_size, num_steps)).at[:, odd_steps].set(-1)
    trajectory_labels = white_perspective_negation * jnp.expand_dims(get_winners(trajectories), 1)
    num_examples = batch_size * num_steps
    states = jnp.reshape(trajectories, (num_examples,) + state_shape)
    state_labels = jnp.reshape(trajectory_labels, (num_examples,))
    return states, state_labels
