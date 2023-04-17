"""Tests for the data module."""

import chex
import jax
import jax.numpy as jnp
import numpy as np

import gojax
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

    def test_sample_game_data_throws_error_on_max_hypo_steps_less_than_one(
            self):
        """Passing max_hypo_steps less than 1 should throw an error."""
        batch_size = 1
        traj_len = 8
        max_hypo_steps = 0
        min_game_len = 4
        traced_trajectories = _make_traced_trajectories(
            batch_size=batch_size,
            traj_len=traj_len,
            min_game_len=min_game_len)
        rng_key = jax.random.PRNGKey(42)

        with self.assertRaises(ValueError):
            data.sample_game_data(traced_trajectories, rng_key, max_hypo_steps)

    def test_sample_game_data_does_not_sample_end_states_beyond_first_terminal_state(
            self):
        """We test this by sampling a lot of data."""
        batch_size = 512
        traj_len = 8
        max_hypo_steps = 4
        min_game_len = 4
        traced_trajectories = _make_traced_trajectories(
            batch_size=batch_size,
            traj_len=traj_len,
            min_game_len=min_game_len)
        rng_key = jax.random.PRNGKey(42)

        game_data = data.sample_game_data(traced_trajectories, rng_key,
                                          max_hypo_steps)

        # Check that the end states are not beyond the first terminal state.
        game_ended = nt_utils.unflatten_first_dim(
            gojax.get_ended(
                nt_utils.flatten_first_two_dims(
                    traced_trajectories.nt_states)), batch_size, traj_len)
        chex.assert_shape(game_ended, (batch_size, traj_len))
        first_terminal_indices = jnp.sum(~game_ended, axis=1) + 1
        chex.assert_rank(game_data.end_states, 4)
        end_state_trace_indices = jnp.sum(
            game_data.end_states[:, gojax.BLACK_CHANNEL_INDEX], axis=(1, 2))
        chex.assert_equal_shape(
            [first_terminal_indices, end_state_trace_indices])
        # Check that the end state trace indices are less than or equal to the
        # first_terminal_indices.
        np.testing.assert_array_less(end_state_trace_indices,
                                     first_terminal_indices + 1)

    def test_sample_game_data_start_state_indices_are_less_than_end_state_indices(
            self):
        """We test this by sampling a lot of data."""
        batch_size = 512
        traj_len = 8
        max_hypo_steps = 4
        min_game_len = 4
        traced_trajectories = _make_traced_trajectories(
            batch_size=batch_size,
            traj_len=traj_len,
            min_game_len=min_game_len)
        rng_key = jax.random.PRNGKey(42)

        game_data = data.sample_game_data(traced_trajectories, rng_key,
                                          max_hypo_steps)

        start_state_trace_indices = jnp.sum(
            game_data.start_states[:, gojax.BLACK_CHANNEL_INDEX], axis=(1, 2))
        end_state_trace_indices = jnp.sum(
            game_data.end_states[:, gojax.BLACK_CHANNEL_INDEX], axis=(1, 2))
        np.testing.assert_array_less(start_state_trace_indices,
                                     end_state_trace_indices)

    def test_sample_game_data_num_non_negative_actions_is_at_most_max_hypo_steps(
            self):
        """We test this by sampling a lot of data."""
        batch_size = 512
        traj_len = 8
        max_hypo_steps = 4
        min_game_len = 4
        traced_trajectories = _make_traced_trajectories(
            batch_size=batch_size,
            traj_len=traj_len,
            min_game_len=min_game_len)
        rng_key = jax.random.PRNGKey(42)

        game_data = data.sample_game_data(traced_trajectories, rng_key,
                                          max_hypo_steps)

        num_non_negative_actions = jnp.sum(game_data.nk_actions >= 0, axis=1)
        np.testing.assert_array_less(
            num_non_negative_actions,
            jnp.full_like(num_non_negative_actions,
                          fill_value=max_hypo_steps + 1))

    def test_sample_game_data_num_non_negative_actions_is_equal_to_end_states_indices_minus_start_state_indices(
            self):
        """We test this by sampling a lot of data."""
        batch_size = 512
        traj_len = 8
        max_hypo_steps = 4
        min_game_len = 4
        traced_trajectories = _make_traced_trajectories(
            batch_size=batch_size,
            traj_len=traj_len,
            min_game_len=min_game_len)
        rng_key = jax.random.PRNGKey(42)

        game_data = data.sample_game_data(traced_trajectories, rng_key,
                                          max_hypo_steps)

        num_non_negative_actions = jnp.sum(game_data.nk_actions >= 0, axis=1)
        start_state_trace_indices = jnp.sum(
            game_data.start_states[:, gojax.BLACK_CHANNEL_INDEX], axis=(1, 2))
        end_state_trace_indices = jnp.sum(
            game_data.end_states[:, gojax.BLACK_CHANNEL_INDEX], axis=(1, 2))
        np.testing.assert_array_equal(
            end_state_trace_indices - start_state_trace_indices,
            num_non_negative_actions)

    def test_sample_game_data_samples_consecutive_states_with_1_max_hypo_steps(
            self):
        """We test this by sampling a lot of data."""
        batch_size = 512
        traj_len = 8
        max_hypo_steps = 1
        min_game_len = 4
        traced_trajectories = _make_traced_trajectories(
            batch_size=batch_size,
            traj_len=traj_len,
            min_game_len=min_game_len)
        rng_key = jax.random.PRNGKey(42)

        game_data = data.sample_game_data(traced_trajectories, rng_key,
                                          max_hypo_steps)

        # Check that the start and end states are consecutive.
        start_state_trace_indices = jnp.sum(
            game_data.start_states[:, gojax.BLACK_CHANNEL_INDEX], axis=(1, 2))
        end_state_trace_indices = jnp.sum(
            game_data.end_states[:, gojax.BLACK_CHANNEL_INDEX], axis=(1, 2))
        np.testing.assert_array_equal(start_state_trace_indices + 1,
                                      end_state_trace_indices)

    def test_sample_game_data_start_player_final_areas_is_first_end_state_areas(
            self):
        """Final areas should match the computed areas of the end state."""
        nt_states = nt_utils.unflatten_first_dim(
            gojax.decode_states("""
                                _ _ _
                                _ _ _
                                _ _ _

                                B _ B
                                _ B _
                                _ W _
                                END=T

                                _ _ _
                                _ W _
                                _ _ _
                                END=T
                                """), 1, 3)
        trajectories = game.Trajectories(nt_states=nt_states,
                                         nt_actions=jnp.array([[4, 1, 0, 0]]))
        game_data = data.sample_game_data(trajectories,
                                          jax.random.PRNGKey(42),
                                          max_hypo_steps=1)
        np.testing.assert_array_equal(
            game_data.start_player_final_areas,
            gojax.compute_areas(
                gojax.decode_states("""
                                    B _ B
                                    _ B _
                                    _ W _
                        """)))

    def test_sample_game_data_start_player_final_areas_is_first_end_state_areas_inverted(
            self):
        """Final areas should match the computed areas of the end state."""
        nt_states = nt_utils.unflatten_first_dim(
            gojax.decode_states("""
                                _ _ _
                                _ _ _
                                _ _ _
                                TURN=W

                                B _ B
                                _ B _
                                _ W _
                                END=T

                                _ _ _
                                _ W _
                                _ _ _
                                END=T
                                """), 1, 3)
        trajectories = game.Trajectories(nt_states=nt_states,
                                         nt_actions=jnp.array([[4, 1, 0, 0]]))
        game_data = data.sample_game_data(trajectories,
                                          jax.random.PRNGKey(42),
                                          max_hypo_steps=1)
        np.testing.assert_array_equal(
            game_data.start_player_final_areas,
            gojax.compute_areas(
                gojax.decode_states("""
                                    W _ W
                                    _ W _
                                    _ B _
                        """)))

    def test_sample_game_data_end_player_final_areas_is_first_end_state_areas_inverted(
            self):
        """Final areas should match the computed areas of the end state."""
        nt_states = nt_utils.unflatten_first_dim(
            gojax.decode_states("""
                                _ _ _
                                _ _ _
                                _ _ _

                                B _ B
                                _ B _
                                _ W _
                                END=T;TURN=W

                                _ _ _
                                _ W _
                                _ _ _
                                END=T
                                """), 1, 3)
        trajectories = game.Trajectories(nt_states=nt_states,
                                         nt_actions=jnp.array([[4, 1, 0, 0]]))
        game_data = data.sample_game_data(trajectories,
                                          jax.random.PRNGKey(42),
                                          max_hypo_steps=1)
        np.testing.assert_array_equal(
            game_data.end_player_final_areas,
            gojax.compute_areas(
                gojax.decode_states("""
                                    W _ W
                                    _ W _
                                    _ B _
                        """)))

    def test_sample_game_data_both_player_final_areas_can_match(self):
        """Final areas should match the computed areas of the end state."""
        nt_states = nt_utils.unflatten_first_dim(
            gojax.decode_states("""
                                _ _ _
                                _ _ _
                                _ _ _

                                _ _ _
                                _ B _
                                _ _ _

                                B _ B
                                _ B _
                                _ W _
                                END=T

                                _ _ _
                                _ W _
                                _ _ _
                                END=T
                                """), 1, 4)
        trajectories = game.Trajectories(nt_states=nt_states,
                                         nt_actions=jnp.array([[4, 1, 0, 0]]))
        game_data = data.sample_game_data(trajectories,
                                          jax.random.PRNGKey(42),
                                          max_hypo_steps=2)
        np.testing.assert_array_equal(
            game_data.start_player_final_areas,
            gojax.compute_areas(
                gojax.decode_states("""
                                    B _ B
                                    _ B _
                                    _ W _
                        """)))
        np.testing.assert_array_equal(
            game_data.end_player_final_areas,
            gojax.compute_areas(
                gojax.decode_states("""
                                    B _ B
                                    _ B _
                                    _ W _
                        """)))

    def test_sample_game_data_on_traced_trajectories_matches_golden(self):
        """Test fixed same game data."""
        batch_size = 8
        traj_len = 8
        max_hypo_steps = 2
        min_game_len = 4
        traced_trajectories = _make_traced_trajectories(
            batch_size=batch_size,
            traj_len=traj_len,
            min_game_len=min_game_len)
        rng_key = jax.random.PRNGKey(42)

        game_data = data.sample_game_data(traced_trajectories, rng_key,
                                          max_hypo_steps)
        np.testing.assert_array_equal(
            game_data.start_player_final_areas,
            gojax.compute_areas(
                gojax.decode_states("""
                                    B B B B B
                                    B B B _ _
                                    _ _ _ _ _
                                    _ _ _ _ _
                                    _ _ _ _ _

                                    B B B B B
                                    B B B _ _
                                    _ _ _ _ _
                                    _ _ _ _ _
                                    _ _ _ _ _

                                    B B B B B
                                    B B B _ _
                                    _ _ _ _ _
                                    _ _ _ _ _
                                    _ _ _ _ _

                                    B B B B B
                                    B B B _ _
                                    _ _ _ _ _
                                    _ _ _ _ _
                                    _ _ _ _ _

                                    B B B B B
                                    B B B _ _
                                    _ _ _ _ _
                                    _ _ _ _ _
                                    _ _ _ _ _

                                    B B B B B
                                    B B B _ _
                                    _ _ _ _ _
                                    _ _ _ _ _
                                    _ _ _ _ _

                                    B B B B B
                                    B B B _ _
                                    _ _ _ _ _
                                    _ _ _ _ _
                                    _ _ _ _ _

                                    B B B B B
                                    B B B _ _
                                    _ _ _ _ _
                                    _ _ _ _ _
                                    _ _ _ _ _
                                    """)))

        np.testing.assert_array_equal(
            game_data.end_player_final_areas,
            gojax.compute_areas(
                gojax.decode_states("""
                                    B B B B B
                                    B B B _ _
                                    _ _ _ _ _
                                    _ _ _ _ _
                                    _ _ _ _ _

                                    B B B B B
                                    B B B _ _
                                    _ _ _ _ _
                                    _ _ _ _ _
                                    _ _ _ _ _

                                    B B B B B
                                    B B B _ _
                                    _ _ _ _ _
                                    _ _ _ _ _
                                    _ _ _ _ _

                                    B B B B B
                                    B B B _ _
                                    _ _ _ _ _
                                    _ _ _ _ _
                                    _ _ _ _ _

                                    B B B B B
                                    B B B _ _
                                    _ _ _ _ _
                                    _ _ _ _ _
                                    _ _ _ _ _

                                    B B B B B
                                    B B B _ _
                                    _ _ _ _ _
                                    _ _ _ _ _
                                    _ _ _ _ _

                                    B B B B B
                                    B B B _ _
                                    _ _ _ _ _
                                    _ _ _ _ _
                                    _ _ _ _ _

                                    B B B B B
                                    B B B _ _
                                    _ _ _ _ _
                                    _ _ _ _ _
                                    _ _ _ _ _
                                    """)))
        np.testing.assert_array_equal(
            game_data.nk_actions,
            jnp.array([[0, 1], [0, 1], [3, -1], [0, -1], [1, -1], [3, 4],
                       [5, -1], [2, 3]]))
        np.testing.assert_array_equal(
            game_data.start_states,
            gojax.decode_states("""
                                B _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 

                                B _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 

                                B B B B _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 

                                B _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 

                                B B _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 

                                B B B B _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 

                                B B B B B 
                                B _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 

                                B B B _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                """),
            gojax.encode_states(game_data.start_states))
        np.testing.assert_array_equal(
            game_data.end_states,
            gojax.decode_states("""
                                B B B _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 

                                B B B _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 

                                B B B B B 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                END=T

                                B B _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 

                                B B B _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 

                                B B B B B 
                                B _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 

                                B B B B B 
                                B B _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                END=T
                                
                                B B B B B 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                _ _ _ _ _ 
                                """),
            gojax.encode_states(game_data.end_states))
