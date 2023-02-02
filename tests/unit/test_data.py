"""Tests for the data module."""

import chex
import gojax
import jax
import jax.numpy as jnp
import numpy as np

from muzero_gojax import data, game, nt_utils


def _make_traced_trajectories(batch_size: int, traj_len: int,
                              min_game_len: int) -> game.Trajectories:
    """Generates trajectories for tracing samples.
    
    The number of pieces on the states, and the actions are equal to
    their index in the trajectory. The last n states are terminal states where n
    is sampled uniformly from [traj_len - min_game_len, traj_len].

    The game length is defined as the number of non-terminal states.

    The board size is 5x5.

    Args:
        batch_size: Batch size.
        traj_len: Trajectory length.
        min_game_len: Minimum game length.

    Returns:
        Trajectories.
    """
    if min_game_len > traj_len:
        raise ValueError('min_game_len must be <= traj_len')
    board_size = 5
    new_trajectories = game.new_trajectories(board_size=board_size,
                                             batch_size=batch_size,
                                             trajectory_length=traj_len)
    nt_states = new_trajectories.nt_states
    for i in range(traj_len):
        col = i % board_size
        row = i // board_size
        nt_states = nt_states.at[:, i:, gojax.BLACK_CHANNEL_INDEX, row,
                                 col].set(True)
    rng_key = jax.random.PRNGKey(42)
    end_indices = jax.random.randint(rng_key,
                                     shape=(batch_size, ),
                                     minval=traj_len - min_game_len - 1,
                                     maxval=traj_len)
    # Terminate all states starting from the end_indices.
    curr_end_indices = end_indices
    for i in range(traj_len):
        nt_states = nt_states.at[jnp.arange(batch_size), curr_end_indices,
                                 gojax.END_CHANNEL_INDEX].set(True)
        curr_end_indices = jnp.minimum(
            jnp.full_like(curr_end_indices, traj_len), curr_end_indices + 1)
    nt_actions = jnp.repeat(jnp.expand_dims(jnp.arange(
        traj_len, dtype=new_trajectories.nt_actions.dtype),
                                            axis=0),
                            batch_size,
                            axis=0)
    chex.assert_equal_shape([nt_actions, new_trajectories.nt_actions])
    return game.Trajectories(nt_states=nt_states, nt_actions=nt_actions)


class DataTestCase(chex.TestCase):
    """Tests data.py"""

    def test_sample_game_data_does_not_sample_end_states_beyond_first_terminal_state(
            self):
        """We test this by sampling a lot of data."""
        batch_size = 512
        traj_len = 8
        traced_trajectories = _make_traced_trajectories(batch_size=batch_size,
                                                        traj_len=traj_len,
                                                        min_game_len=4)
        rng_key = jax.random.PRNGKey(42)
        max_hypothetical_steps = 4
        game_data = data.sample_game_data(traced_trajectories, rng_key,
                                          max_hypothetical_steps)
        # Check that the end states are not beyond the first terminal state.
        game_ended = nt_utils.unflatten_first_dim(
            gojax.get_ended(
                nt_utils.flatten_first_two_dims(
                    traced_trajectories.nt_states)), batch_size, traj_len)
        chex.assert_shape(game_ended, (batch_size, traj_len))
        end_indices = jnp.sum(~game_ended, axis=1)
        chex.assert_rank(game_data.end_states, 4)
        end_state_trace_indices = jnp.sum(
            game_data.end_states[:, gojax.BLACK_CHANNEL_INDEX], axis=(1, 2))
        chex.assert_equal_shape([end_indices, end_state_trace_indices])
        # Check that the end state trace indices are less than or equal to the
        # end_indices.
        np.testing.assert_array_less(end_state_trace_indices, end_indices + 1)
