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
    embed_model, _, policy_model, _ = model_fn.apply
    logits = policy_model(params, rng_key, embed_model(params, rng_key, states))
    states = gojax.next_states(states, gojax.sample_non_occupied_actions(states, logits, rng_key))
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


def self_play(model_fn, batch_size, board_size, num_steps, params, rng_key):
    # pylint: disable=too-many-arguments
    """
    Simulates a batch of trajectories made from playing the model against itself.

    :param model_fn: a model function that takes in a batch of Go states and parameters and
    outputs a batch of action
    probabilities for each state.
    :param batch_size: N.
    :param board_size: B.
    :param num_steps: number of steps to take.
    :param params: the model parameters.
    :param rng_key: RNG key used to seed the randomness of the self play.
    :return: an N x T x C x B x B boolean array.
    """
    return lax.fori_loop(0, num_steps - 1,
                         jax.tree_util.Partial(update_trajectories, model_fn, params, rng_key),
                         new_trajectories(board_size, batch_size, num_steps))


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


def get_actions_and_labels(trajectories):
    """
    Extracts action indices and game winners from the trajectories.

    The label ({-1, 0, 1}) for the corresponding state represents the winner of the outcome of
    that state's trajectory.

    :param trajectories: An N x T x C x B x B boolean array.
    :return: trajectories, an N x T non-negative integer array representing action indices,
    and an N x T integer {-1, 0, 1} array representing game winners.
    """
    batch_size, num_steps = trajectories.shape[:2]
    state_shape = trajectories.shape[2:]
    odd_steps = jnp.arange(num_steps // 2) * 2 + 1
    white_perspective_negation = jnp.ones((batch_size, num_steps)).at[:, odd_steps].set(-1)
    game_winners = white_perspective_negation * jnp.expand_dims(get_winners(trajectories), 1)
    num_examples = batch_size * num_steps
    states = jnp.reshape(trajectories, (num_examples,) + state_shape)
    occupied_spaces = gojax.get_occupied_spaces(states)
    indicator_actions = jnp.logical_xor(occupied_spaces, jnp.roll(occupied_spaces, -1, axis=0))
    action_indices = jnp.reshape(gojax.action_indicators_to_indices(indicator_actions),
                                 (batch_size, num_steps))
    return action_indices, game_winners
