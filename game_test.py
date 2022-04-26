import unittest

import chex
import haiku as hk
import jax.numpy as jnp
import jax.random
import numpy as np
from gojax import go

import main
import game


def _parse_state_string_buffer(state_string_buffer, trajectory, turn):
    state_string = ''.join(state_string_buffer)
    trajectory.append(go.decode_state(state_string, turn))
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
    def test_random_self_play_3x3_42rng(self):
        go_model = hk.transform(lambda states: main.RandomGoModel()(states))
        trajectories = simulation.self_play(go_model, params={}, batch_size=1, board_size=3, max_num_steps=6,
                                            rng_key=jax.random.PRNGKey(42))
        expected_trajectories = _read_trajectory('tests/random_self_play_3x3_42rng_expected_trajectory.txt')
        np.testing.assert_array_equal(trajectories, expected_trajectories)


if __name__ == '__main__':
    unittest.main()
