"""Tests game.py."""
# pylint: disable=missing-function-docstring,no-self-use,unnecessary-lambda

import unittest

import chex
import gojax
import haiku as hk
import jax.numpy as jnp
import jax.random
import numpy as np

import game
import models


def _parse_state_string_buffer(state_string_buffer, turn, previous_state=None):
    state_string = ''.join(state_string_buffer)
    state = gojax.decode_state(state_string, turn)
    if previous_state is not None:
        state = state.at[:, gojax.PASS_CHANNEL_INDEX].set(jnp.alltrue(
            gojax.get_occupied_spaces(state) == gojax.get_occupied_spaces(previous_state)))
    return state


def _read_trajectory(filename):
    trajectory = []
    state_string_buffer = []
    turn = False
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            if line.strip():
                state_string_buffer.append(line)
            else:
                trajectory.append(
                    _parse_state_string_buffer(state_string_buffer, turn,
                                               trajectory[-1] if trajectory else None))
                state_string_buffer.clear()
                turn = not turn
    if state_string_buffer:
        trajectory.append(_parse_state_string_buffer(state_string_buffer, turn,
                                                     trajectory[-1] if trajectory else None))
        state_string_buffer.clear()
    return jnp.stack(trajectory, axis=1)


def _get_trajectory_pretty_string(trajectories, index=0):
    pretty_trajectory_str = '\n'.join(
        map(lambda state: gojax.get_pretty_string(state), trajectories[index]))
    return pretty_trajectory_str


class GameTestCase(chex.TestCase):
    """Tests game.py."""

    def test_new_trajectories(self):
        new_trajectories = game.new_trajectories(board_size=3, batch_size=2, max_num_steps=9)
        chex.assert_shape(new_trajectories, (2, 9, 6, 3, 3))
        np.testing.assert_array_equal(new_trajectories, jnp.zeros_like(new_trajectories))

    def test_read_sample_trajectory(self):
        sample_trajectory = _read_trajectory('tests/sample_trajectory.txt')
        chex.assert_shape(sample_trajectory, (1, 3, 6, 3, 3))
        np.testing.assert_array_equal(sample_trajectory[:, 0],
                                      gojax.decode_state("""
                                                        _ _ _
                                                        _ _ _
                                                        _ _ _
                                                        """, turn=gojax.BLACKS_TURN))
        np.testing.assert_array_equal(sample_trajectory[:, 1],
                                      gojax.decode_state("""
                                                        _ _ _
                                                        _ B _
                                                        _ _ _
                                                        """, turn=gojax.WHITES_TURN))

        np.testing.assert_array_equal(sample_trajectory[:, 2],
                                      gojax.decode_state("""
                                                        _ _ _
                                                        _ B _
                                                        _ _ _
                                                        """, turn=gojax.BLACKS_TURN,
                                                         passed=True))

    def test_random_sample_next_states_3x3_42rng(self):
        go_model = hk.transform(lambda states: models.RandomGoModel()(states))
        # We use the same RNG key that would be used in the update_trajectories function.
        next_states = game.sample_next_states(go_model, params={},
                                              rng_key=jax.random.fold_in(jax.random.PRNGKey(42), 0),
                                              states=gojax.new_states(board_size=3))
        expected_next_states = gojax.decode_state("""
                                        _ _ _
                                        _ _ _
                                        _ _ B
                                        """, turn=gojax.WHITES_TURN)
        np.testing.assert_array_equal(next_states, expected_next_states)

    def test_update_trajectories_step_0(self):
        go_model = hk.transform(lambda states: models.RandomGoModel()(states))
        trajectories = game.new_trajectories(board_size=3, batch_size=1, max_num_steps=6)
        updated_trajectories = game.update_trajectories(go_model, params={},
                                                        rng_key=jax.random.PRNGKey(42), step=0,
                                                        trajectories=trajectories)
        np.testing.assert_array_equal(updated_trajectories[:, 0],
                                      jnp.zeros_like(updated_trajectories[:, 0]))
        np.testing.assert_array_equal(updated_trajectories[:, 1],
                                      gojax.decode_state("""
                                                        _ _ _
                                                        _ _ _
                                                        _ _ B
                                                        """,
                                                         turn=gojax.WHITES_TURN))

    def test_update_trajectories_step_1(self):
        go_model = hk.transform(lambda states: models.RandomGoModel()(states))
        trajectories = game.new_trajectories(board_size=3, batch_size=1, max_num_steps=6)
        updated_trajectories = game.update_trajectories(go_model, params={},
                                                        rng_key=jax.random.PRNGKey(42), step=1,
                                                        trajectories=trajectories)
        np.testing.assert_array_equal(updated_trajectories[:, 0],
                                      jnp.zeros_like(updated_trajectories[:, 0]))
        np.testing.assert_array_equal(updated_trajectories[:, 1],
                                      jnp.zeros_like(updated_trajectories[:, 1]))
        np.testing.assert_array_equal(updated_trajectories[:, 2],
                                      gojax.decode_state("""
                                                        _ _ _
                                                        _ _ _
                                                        _ _ B
                                                        """,
                                                         turn=gojax.WHITES_TURN))

    @chex.variants(with_jit=True, without_jit=True)
    def test_random_self_play_3x3_42rng(self):
        go_model = hk.transform(lambda states: models.RandomGoModel()(states))
        self_play_fn = self.variant(game.self_play, static_argnums=(0, 2, 3, 4))
        trajectories = self_play_fn(go_model, params={}, batch_size=1, board_size=3,
                                    max_num_steps=6, rng_key=jax.random.PRNGKey(42))
        expected_trajectories = _read_trajectory(
            'tests/random_self_play_3x3_42rng_expected_trajectory.txt')
        pretty_trajectory_str = _get_trajectory_pretty_string(trajectories)
        np.testing.assert_array_equal(trajectories, expected_trajectories, pretty_trajectory_str)

    def test_get_winners_one_tie_one_winning_one_winner(self):
        trajectories = game.new_trajectories(board_size=3, batch_size=3, max_num_steps=2)
        trajectories = trajectories.at[:1, 1].set(gojax.decode_state("""
                                                                    _ _ _
                                                                    _ _ _
                                                                    _ _ _
                                                                    """, ended=False))
        trajectories = trajectories.at[1:2, 1].set(gojax.decode_state("""
                                                                    _ _ _
                                                                    _ B _
                                                                    _ _ _
                                                                    """, ended=False))
        trajectories = trajectories.at[2:3, 1].set(gojax.decode_state("""
                                                                    _ _ _
                                                                    _ B _
                                                                    _ _ _
                                                                    """, ended=True))
        winners = game.get_winners(trajectories)
        np.testing.assert_array_equal(winners, [0, 1, 1])

    def test_trajectories_to_dataset_with_sample_trajectory(self):
        sample_trajectory = _read_trajectory('tests/sample_trajectory.txt')
        states, labels = game.trajectories_to_dataset(sample_trajectory)
        np.testing.assert_array_equal(states,
                                      jnp.concatenate((gojax.decode_state("""
                                                                        _ _ _
                                                                        _ _ _
                                                                        _ _ _
                                                                        """),
                                                       gojax.decode_state("""
                                                                        _ _ _
                                                                        _ B _
                                                                        _ _ _
                                                                        """,
                                                                          turn=gojax.WHITES_TURN),
                                                       gojax.decode_state("""
                                                                       _ _ _
                                                                       _ B _
                                                                       _ _ _
                                                                       """, passed=True))))

        np.testing.assert_array_equal(labels, [1, -1, 1])


if __name__ == '__main__':
    unittest.main()
