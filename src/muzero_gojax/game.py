"""Manages the model generation of Go games."""

from typing import Tuple

import absl.flags
import gojax
import haiku as hk
import jax.nn
import jax.random
import jax.tree_util
import optax
from jax import lax
from jax import numpy as jnp


def sample_actions_and_next_states(go_model: hk.MultiTransformed, params: optax.Params,
                                   rng_key: jax.random.KeyArray, states: jnp.ndarray) -> Tuple[
    jnp.ndarray, jnp.ndarray]:
    """
    Simulates the next states of the Go game played out by the given model.

    :param go_model: a model function that takes in a batch of Go states and parameters and
    outputs a batch of action
    probabilities for each state.
    :param params: the model parameters.
    :param rng_key: RNG key used to seed the randomness of the simulation.
    :param states: a batch array of N Go games.
    :return: an N-dimensional integer vector and a N x C x B x B boolean array of Go games.
    """
    logits = get_policy_logits(go_model, params, states, rng_key)
    actions = jax.random.categorical(rng_key, logits)
    next_states = gojax.next_states(states, actions)
    return actions, next_states


def get_policy_logits(go_model: hk.MultiTransformed, params: optax.Params, states: jnp.ndarray,
                      rng_key: jax.random.KeyArray) -> jnp.ndarray:
    """Gets the policy logits from the model. """
    embed_model, _, policy_model, _ = go_model.apply
    return policy_model(params, rng_key, embed_model(params, rng_key, states))


def new_traj_states(board_size: int, batch_size: int, max_num_steps: int) -> jnp.ndarray:
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


def update_trajectories(go_model: hk.MultiTransformed, params: optax.Params,
                        rng_key: jax.random.KeyArray, step: int, trajectories: dict) -> jnp.ndarray:
    """
    Updates the trajectory array for time step `step + 1`.

    :param go_model: a model function that takes in a batch of Go states and parameters and
    outputs a batch of action
    probabilities for each state.
    :param params: the model parameters.
    :param rng_key: RNG key which is salted by the time step.
    :param step: the current time step of the trajectory.
    :param trajectories: A dictionary containing
      * nt_states: an N x T x C x B x B boolean array
      * nt_actions: an N x T integer array
    :return: an N x T x C x B x B boolean array
    """
    actions, next_states = sample_actions_and_next_states(go_model, params,
                                                          jax.random.fold_in(rng_key, step),
                                                          trajectories['nt_states'][:, step])
    trajectories['nt_actions'] = trajectories['nt_actions'].at[:, step].set(actions)
    trajectories['nt_states'] = trajectories['nt_states'].at[:, step + 1].set(next_states)
    return trajectories


def self_play(absl_flags: absl.flags.FlagValues, go_model: hk.MultiTransformed,
              params: optax.Params, rng_key: jax.random.KeyArray) -> dict:
    """
    Simulates a batch of trajectories made from playing the model against itself.

    :param absl_flags: Abseil flags.
    :param go_model: a model function that takes in a batch of Go states and parameters and
    outputs a batch of action
    probabilities for each state.
    :param params: the model parameters.
    :param rng_key: RNG key used to seed the randomness of the self play.
    :return: an N x T x C x B x B boolean array.
    """
    # We iterate max_num_steps - 1 times because we start updating the second column of the
    # trajectories array, not the first.
    return lax.fori_loop(0, absl_flags.max_num_steps - 1,
                         jax.tree_util.Partial(update_trajectories, go_model, params, rng_key), {
                             'nt_states': new_traj_states(absl_flags.board_size,
                                                          absl_flags.batch_size,
                                                          absl_flags.max_num_steps),
                             'nt_actions': jnp.full(
                                 (absl_flags.batch_size, absl_flags.max_num_steps), fill_value=-1,
                                 dtype='uint16')
                         })


def get_winners(trajectories: jnp.ndarray) -> jnp.ndarray:
    """
    Gets the winner for each trajectory.

    1 = black won
    0 = tie
    -1 = white won

    :param trajectories: an N x T x C x B x B boolean array.
    :return: a boolean array of length N.
    """
    return gojax.compute_winning(trajectories[:, -1])


def get_labels(nt_states: jnp.ndarray) -> jnp.ndarray:
    """
    Extracts action indices and game winners from the trajectories.

    The label ({-1, 0, 1}) for the corresponding state represents the winner of the outcome of
    that state's trajectory.

    :param nt_states: An N x T x C x B x B boolean array of trajectory states.
    :return: An N x T integer {-1, 0, 1} array representing whether the player whose turn it is on the
    corresponding state ended up winning, tying, or losing. The last action is undefined and has no
    meaning because it is associated with the last state where no action was taken.
    """
    batch_size, num_steps = nt_states.shape[:2]
    state_shape = nt_states.shape[2:]
    odd_steps = jnp.arange(num_steps // 2) * 2 + 1
    white_perspective_negation = jnp.ones((batch_size, num_steps), dtype='int8').at[:,
                                 odd_steps].set(-1)
    return white_perspective_negation * jnp.expand_dims(get_winners(nt_states), 1)
