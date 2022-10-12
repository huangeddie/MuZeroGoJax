"""Manages the model generation of Go games."""
from typing import NamedTuple
from typing import Tuple

import gojax
import haiku as hk
import jax.nn
import jax.random
import jax.tree_util
import optax
from absl import flags
from jax import lax
from jax import numpy as jnp

from muzero_gojax import models

FLAGS = flags.FLAGS


class Trajectories(NamedTuple):
    """A series of Go states and actions."""
    # [N, T, C, B, B] boolean tensor.
    nt_states: jnp.ndarray = None
    # [N, T] integer tensor.
    nt_actions: jnp.ndarray = None


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
    actions = jax.random.categorical(rng_key, logits).astype('uint16')
    next_states = gojax.next_states(states, actions)
    return actions, next_states


def get_policy_logits(go_model: hk.MultiTransformed, params: optax.Params, states: jnp.ndarray,
                      rng_key: jax.random.KeyArray) -> jnp.ndarray:
    """Gets the policy logits from the model. """
    embed_model = go_model.apply[models.EMBED_INDEX]
    policy_model = go_model.apply[models.POLICY_INDEX]
    return policy_model(params, rng_key, embed_model(params, rng_key, states))


def new_trajectories(board_size: int, batch_size: int, trajectory_length: int) -> Trajectories:
    """
    Creates an empty array of Go game trajectories.

    :param board_size: B.
    :param batch_size: N.
    :param trajectory_length: T.
    :return: an N x T x C x B x B boolean array, where the third dimension (C) contains
    information about the Go game
    state.
    """
    empty_trajectories = jnp.repeat(
        jnp.expand_dims(gojax.new_states(board_size, batch_size), axis=1), trajectory_length, 1)
    return Trajectories(nt_states=empty_trajectories,
                        nt_actions=jnp.full((batch_size, trajectory_length), fill_value=-1,
                                            dtype='uint16'))


def update_trajectories(go_model: hk.MultiTransformed, params: optax.Params,
                        rng_key: jax.random.KeyArray, step: int,
                        trajectories: Trajectories) -> Trajectories:
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
                                                          trajectories.nt_states[:, step])
    trajectories = trajectories._replace(
        nt_actions=trajectories.nt_actions.at[:, step].set(actions))
    trajectories = trajectories._replace(
        nt_states=trajectories.nt_states.at[:, step + 1].set(next_states))
    return trajectories


def self_play(empty_trajectories: Trajectories, go_model: hk.MultiTransformed, params: optax.Params,
              rng_key: jax.random.KeyArray) -> Trajectories:
    """
    Simulates a batch of trajectories made from playing the model against itself.

    :param empty_trajectories: Empty trajectories to fill.
    :param go_model: a model function that takes in a batch of Go states and parameters and
    outputs a batch of action
    probabilities for each state.
    :param params: the model parameters.
    :param rng_key: RNG key used to seed the randomness of the self play.
    :return: an N x T x C x B x B boolean array.
    """
    # We iterate trajectory_length - 1 times because we start updating the second column of the
    # trajectories array, not the first.
    return lax.fori_loop(0, empty_trajectories.nt_states.shape[1] - 1,
                         jax.tree_util.Partial(update_trajectories, go_model, params, rng_key),
                         empty_trajectories)


def get_winners(nt_states: jnp.ndarray) -> jnp.ndarray:
    """
    Gets the winner for each trajectory.

    1 = black won
    0 = tie
    -1 = white won

    :param nt_states: an N x T x C x B x B boolean array.
    :return: a boolean array of length N.
    """
    return gojax.compute_winning(nt_states[:, -1])


def get_labels(nt_states: jnp.ndarray) -> jnp.ndarray:
    """
    Extracts action indices and game winners from the trajectories.

    The label ({-1, 0, 1}) for the corresponding state represents the winner of the outcome of
    that state's trajectory.

    :param nt_states: An N x T x C x B x B boolean array of trajectory states.
    :return: An N x T integer {-1, 0, 1} array representing whether the player whose turn it is on
    the corresponding state ended up winning, tying, or losing. The last action is undefined and has
    no meaning because it is associated with the last state where no action was taken.
    """
    batch_size, num_steps = nt_states.shape[:2]
    odd_steps = jnp.arange(num_steps // 2) * 2 + 1
    white_perspective_negation = jnp.ones((batch_size, num_steps), dtype='int8').at[:,
                                 odd_steps].set(-1)
    return white_perspective_negation * jnp.expand_dims(get_winners(nt_states), 1)
