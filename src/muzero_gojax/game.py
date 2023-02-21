"""Manages the model generation of Go games."""
import re
from typing import Tuple

import chex
import gojax
import jax.nn
import jax.random
import jax.tree_util
from absl import flags
from jax import lax
from jax import numpy as jnp

from muzero_gojax import models, nt_utils

FLAGS = flags.FLAGS


@chex.dataclass(frozen=True)
class Trajectories:
    """A series of Go states and actions."""
    # N x T x C x B x B boolean tensor.
    nt_states: jnp.ndarray = None
    # N x T integer tensor.
    nt_actions: jnp.ndarray = None


@chex.dataclass(frozen=True)
class GameStats:
    """Data about the game."""
    # TODO: Remove these default values.
    avg_game_length: jnp.ndarray = jnp.array(-1)
    black_wins: jnp.ndarray = jnp.array(-1, dtype='int32')
    ties: jnp.ndarray = jnp.array(-1, dtype='int32')
    white_wins: jnp.ndarray = jnp.array(-1, dtype='int32')
    # The rate at which the actions collide with pieces on the board.
    # This is a sign that the policies are not learning to avoid collisions.
    piece_collision_rate: jnp.ndarray = jnp.array(-1, dtype='float32')
    pass_rate: jnp.ndarray = jnp.array(-1, dtype='float32')


def new_trajectories(board_size: int, batch_size: int,
                     trajectory_length: int) -> Trajectories:
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
        jnp.expand_dims(gojax.new_states(board_size, batch_size), axis=1),
        trajectory_length, 1)
    return Trajectories(nt_states=empty_trajectories,
                        nt_actions=jnp.full((batch_size, trajectory_length),
                                            fill_value=-1,
                                            dtype='uint16'))


def _update_two_player_trajectories(
        black_policy: models.PolicyModel, white_policy: models.PolicyModel,
        rng_key: jax.random.KeyArray, step: int,
        trajectories: Trajectories) -> Trajectories:
    """
    Updates the trajectory array for time step `step + 1`.

    :param black_policy: Black policy model.
    :param white_policy: White policy model.
    :param rng_key: RNG key which is salted by the time step.
    :param step: the current time step of the trajectory.
    :param trajectories: A dictionary containing
      * nt_states: an N x T x C x B x B boolean array
      * nt_actions: an N x T integer array
    :return: an N x T x C x B x B boolean array
    """
    rng_key = jax.random.fold_in(rng_key, step)
    states = trajectories.nt_states[:, step]
    policy_output: models.PolicyOutput = lax.cond(
        step % 2 == 0, jax.tree_util.Partial(black_policy, rng_key),
        jax.tree_util.Partial(white_policy, rng_key), states)
    next_states = gojax.next_states(states, policy_output.sampled_actions)
    trajectories = trajectories.replace(
        nt_actions=trajectories.nt_actions.at[:, step].set(
            policy_output.sampled_actions))
    trajectories = trajectories.replace(
        nt_states=trajectories.nt_states.at[:, step + 1].set(next_states))
    return trajectories


def _get_winners(nt_states: jnp.ndarray) -> jnp.ndarray:
    """
    Gets the winner for each trajectory.

    1 = black won
    0 = tie
    -1 = white won

    :param nt_states: an N x T x C x B x B boolean array.
    :return: a boolean array of length N.
    """
    return gojax.compute_winning(nt_states[:, -1])


def _count_wins(nt_states: jnp.ndarray) -> Tuple[jnp.ndarray]:
    """Counts the total number of black wins, ties, and white wins.

    Args:
        nt_states (jnp.ndarray): A batch of Go trajectories.

    Returns:
        Tuple[jnp.ndarray]: black wins, ties, white wins.
    """
    winners = _get_winners(nt_states)
    return jnp.sum(winners == 1), jnp.sum(winners == 0), jnp.sum(winners == -1)


def get_nt_player_labels(nt_states: jnp.ndarray) -> jnp.ndarray:
    """
    Game winners from the trajectories from the perspective of the player.

    The label ({-1, 0, 1}) for the corresponding state represents the winner of the outcome of
    that state's trajectory.

    :param nt_states: An N x T x C x B x B boolean array of trajectory states.
    :return: An N x T integer {-1, 0, 1} array representing whether the player whose turn it is on
    the corresponding state ended up winning, tying, or losing. The last action is undefined and has
    no meaning because it is associated with the last state where no action was taken.
    """
    batch_size, num_steps = nt_states.shape[:2]
    ones = jnp.ones((batch_size, num_steps), dtype='int8')
    white_perspective_negation = ones.at[:, 1::2].set(-1)
    return white_perspective_negation * jnp.expand_dims(
        _get_winners(nt_states), 1)


def get_game_stats(trajectories: Trajectories) -> GameStats:
    """Gets game statistics from trajectories."""
    nt_states = trajectories.nt_states
    black_wins, ties, white_wins = _count_wins(nt_states)
    game_ended = gojax.get_ended(nt_utils.flatten_first_two_dims(nt_states))
    num_non_terminal_states = jnp.sum(~game_ended)
    avg_game_length = num_non_terminal_states / len(nt_states)
    states = nt_utils.flatten_first_two_dims(nt_states)
    any_pieces = (states[:, gojax.BLACK_CHANNEL_INDEX]
                  | states[:, gojax.WHITE_CHANNEL_INDEX])
    actions = nt_utils.flatten_first_two_dims(trajectories.nt_actions)
    indicator_actions = gojax.action_1d_to_indicator(actions,
                                                     nrows=states.shape[-2],
                                                     ncols=states.shape[-1])
    piece_collision_rate = jnp.sum(
        jnp.sum(indicator_actions & any_pieces, axis=(1, 2)) *
        ~game_ended) / num_non_terminal_states
    board_size = nt_states.shape[-1]
    pass_rate = jnp.sum(
        (actions == jnp.full_like(actions, fill_value=board_size**2))
        & ~game_ended) / num_non_terminal_states
    return GameStats(avg_game_length=avg_game_length,
                     black_wins=black_wins,
                     ties=ties,
                     white_wins=white_wins,
                     piece_collision_rate=piece_collision_rate,
                     pass_rate=pass_rate)


def rotationally_augment_trajectories(
        trajectories: Trajectories) -> Trajectories:
    """
    Divides the batch (0) dimension into four segments and rotates each
    section 90 degrees counter-clockwise times their section index.

    :param trajectories:
    :return: rotationally augmented trajectories.
    """
    nt_states = trajectories.nt_states
    batch_size, trajectory_length = nt_states.shape[:2]
    nrows, ncols = nt_states.shape[-2:]
    nt_indicator_actions = nt_utils.unflatten_first_dim(
        gojax.action_1d_to_indicator(
            nt_utils.flatten_first_two_dims(trajectories.nt_actions), nrows,
            ncols), batch_size, trajectory_length)
    group_size = max(len(nt_states) // 4, 1)
    nt_aug_states = nt_states
    nt_aug_indic_actions = nt_indicator_actions
    for i in range(1, 4):
        if i >= len(nt_states):
            break
        sliced_augmented_states = jnp.rot90(
            nt_aug_states[i * group_size:(i + 1) * group_size],
            k=i,
            axes=(3, 4))
        sliced_augmented_actions = jnp.rot90(
            nt_aug_indic_actions[i * group_size:(i + 1) * group_size],
            k=i,
            axes=(2, 3))
        nt_aug_states = nt_aug_states.at[i * group_size:(i + 1) *
                                         group_size].set(
                                             sliced_augmented_states)
        nt_aug_indic_actions = nt_aug_indic_actions.at[i * group_size:(
            i + 1) * group_size].set(sliced_augmented_actions)

    nt_aug_actions = nt_utils.unflatten_first_dim(
        gojax.action_indicator_to_1d(
            nt_utils.flatten_first_two_dims(nt_aug_indic_actions)), batch_size,
        trajectory_length)
    return trajectories.replace(nt_states=nt_aug_states,
                                nt_actions=nt_aug_actions)


def _pit(a_policy: models.PolicyModel, b_policy: models.PolicyModel,
         empty_trajectories: Trajectories,
         rng_key: jax.random.KeyArray) -> Trajectories:
    # We iterate trajectory_length - 1 times because we start updating the second column of the
    # trajectories array, not the first.
    return lax.fori_loop(
        0, empty_trajectories.nt_states.shape[1] - 1,
        jax.tree_util.Partial(_update_two_player_trajectories, a_policy,
                              b_policy, rng_key), empty_trajectories)


def pit(a_policy: models.PolicyModel, b_policy: models.PolicyModel,
        board_size: int, n_games: int,
        traj_len: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pits models A and B against each other n_games times.

    Equally distributes the number of times A and B are black and white,
    with A getting the one extra time to be black if n_games is odd.

    Args:
        a_policy (jax.tree_util.Partial): A Go policy model.
        b_policy (jax.tree_util.Partial): Another Go policy model.
        board_size (int): Board size.
        n_games (int): Number of games to play.
        traj_len (int): Number of steps per game.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Number of times A won,
            number of ties, number of times B won.
    """
    batch_size = n_games // 2
    a_rng_key, b_rng_key = jax.random.split(jax.random.PRNGKey(42))
    a_starts_traj = _pit(
        a_policy, b_policy,
        new_trajectories(board_size, batch_size, trajectory_length=traj_len),
        a_rng_key)
    b_starts_traj = _pit(
        b_policy, a_policy,
        new_trajectories(board_size, batch_size, trajectory_length=traj_len),
        b_rng_key)
    a_starts_a_wins, a_starts_ties, a_starts_b_wins = _count_wins(
        a_starts_traj.nt_states)
    b_starts_b_wins, b_starts_ties, b_starts_a_wins = _count_wins(
        b_starts_traj.nt_states)
    return (a_starts_a_wins +
            b_starts_a_wins), (a_starts_ties +
                               b_starts_ties), (a_starts_b_wins +
                                                b_starts_b_wins)


def self_play(empty_trajectories: Trajectories,
              policy_model: models.PolicyModel,
              rng_key: jax.random.KeyArray) -> Trajectories:
    """
    Simulates a batch of trajectories made from playing the model against itself.

    :param empty_trajectories: Empty trajectories to fill.
    :param policy: Policy model.
    :param rng_key: RNG key used to seed the randomness of the self play.
    :return: an N x T x C x B x B boolean array.
    """
    # We iterate trajectory_length - 1 times because we start updating the second column of the
    # trajectories array, not the first.
    return _pit(policy_model, policy_model, empty_trajectories, rng_key)


def estimate_elo_rating(opponent_elo: int, wins: int, ties: int,
                        losses: int) -> int:
    """Estimates the Elo rating.
    
    (Opponent's rating + 400 x (wins - losses)) / (total number of games)
    """
    num_games = wins + ties + losses
    return (opponent_elo * num_games + 400 * (wins - losses)) / num_games


@chex.dataclass(frozen=True)
class UserMove:
    """User move data structure for playing against models."""
    row: int
    col: int
    passed: bool
    exit: bool


def _get_user_move(input_fn) -> UserMove:
    cap_letters = 'ABCDEFGHIJKLMNOPQRS'

    user_input = input_fn('Enter action:').lower()
    understood_expression = False
    while not understood_expression:
        row_col_match = re.match(r'\s*(\d+)\s*(\D+)\s*', user_input)
        if row_col_match is not None:
            row = int(row_col_match.group(1))
            col = cap_letters.index(row_col_match.group(2).upper())
            return UserMove(row=row, col=col, passed=False, exit=False)
        pass_match = re.match(r'\s*(pass)\s*', user_input)
        if pass_match is not None:
            return UserMove(row=None, col=None, passed=True, exit=False)
        exit_match = re.match(r'\s*(exit)\s*', user_input)
        if exit_match is not None:
            return UserMove(row=None, col=None, passed=False, exit=True)


def play_against_model(policy: models.PolicyModel,
                       board_size,
                       input_fn=None,
                       play_as_white=False):
    """
    Deploys an interactive terminal to play against the Go model.

    :param go_model: Haiku Go model.
    :param params: Model parameters.
    :param board_size: Board size.
    :return: None.
    """
    print('=' * 80)
    print('Enter move (R C), pass (pass), or exit (exit)')
    print('=' * 80)

    if input_fn is None:
        input_fn = input

    states = gojax.new_states(board_size)
    gojax.print_state(states[0])
    rng_key = jax.random.PRNGKey(seed=42)
    if play_as_white:
        # Get AI's move.
        print('Model thinking...')
        _, rng_key = jax.random.split(rng_key)
        policy_output: models.PolicyOutput = policy(rng_key, states)
        states = gojax.next_states(states, policy_output.sampled_actions)
        gojax.print_state(states[0])
    while not gojax.get_ended(states):
        # Get user's move.
        user_move: UserMove = _get_user_move(input_fn)
        if user_move.exit:
            return
        elif user_move.row is not None and user_move.col is not None:
            action = user_move.row * states.shape[-1] + user_move.col
        elif user_move.passed:
            action = board_size**2
        else:
            raise RuntimeError(f'Invalid user move: {user_move}')
        states = gojax.next_states(states, jnp.array([action]))
        gojax.print_state(states[0])
        if gojax.get_ended(states):
            break

        # Get AI's move.
        print('Model thinking...')
        _, rng_key = jax.random.split(rng_key)
        policy_output: models.PolicyOutput = policy(rng_key, states)
        states = gojax.next_states(states, policy_output.sampled_actions)
        gojax.print_state(states[0])
