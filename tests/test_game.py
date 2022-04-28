import unittest

import chex
import gojax
import haiku as hk
import jax.numpy as jnp
import jax.random
import numpy as np

import game
import models


def _parse_state_string_buffer(state_string_buffer, trajectory, turn):
    state_string = ''.join(state_string_buffer)
    trajectory.append(gojax.decode_state(state_string, turn))
    turn = not turn
    state_string_buffer.clear()
    return turn


def _read_trajectory(filename):
    trajectory = []
    state_string_buffer = []
    turn = False
    with open(filename, 'r') as f:
        for line in f.readlines():
            if line.strip():
                state_string_buffer.append(line)
            else:
                turn = _parse_state_string_buffer(state_string_buffer, trajectory, turn)
    if state_string_buffer:
        _parse_state_string_buffer(state_string_buffer, trajectory, turn)
    return jnp.stack(trajectory, axis=1)


class MyTestCase(chex.TestCase):
    def test_read_sample_trajectory(self):
        sample_trajectory = _read_trajectory('sample_trajectory.txt')
        chex.assert_shape(sample_trajectory, (1, 2, 6, 3, 3))
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

    def test_random_sample_next_states_3x3_42rng(self):
        go_model = hk.transform(lambda states: models.RandomGoModel()(states))
        next_states = game.sample_next_states(go_model, params={}, rng_key=jax.random.PRNGKey(42),
                                              states=gojax.new_states(board_size=3))
        expected_next_states = gojax.decode_state("""
                                        _ _ _
                                        B _ _
                                        _ _ _
                                        """, turn=gojax.WHITES_TURN)
        np.testing.assert_array_equal(next_states, expected_next_states)

    def test_random_self_play_3x3_42rng(self):
        go_model = hk.transform(lambda states: models.RandomGoModel()(states))
        trajectories = game.self_play(go_model, params={}, batch_size=1, board_size=3, max_num_steps=6,
                                      rng_key=jax.random.PRNGKey(42))
        expected_trajectories = _read_trajectory('random_self_play_3x3_42rng_expected_trajectory.txt')
        np.testing.assert_array_equal(trajectories, expected_trajectories)

    def test_new_trajectories(self):
        new_trajectories = game.new_trajectories(board_size=3, batch_size=2, max_num_steps=9)
        chex.assert_shape(new_trajectories, (2, 9, 6, 3, 3))
        np.testing.assert_array_equal(new_trajectories, jnp.zeros_like(new_trajectories))


if __name__ == '__main__':
    unittest.main()
