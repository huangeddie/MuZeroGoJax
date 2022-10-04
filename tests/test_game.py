"""Tests game.py."""
# pylint: disable=missing-function-docstring,duplicate-code
import unittest

import chex
import gojax
import jax.numpy as jnp
import jax.random
import numpy as np

from muzero_gojax import game
from muzero_gojax import main
from muzero_gojax import models


class GameTestCase(chex.TestCase):
    """Tests game.py."""

    def setUp(self):
        self.board_size = 3
        main.FLAGS(
            f'foo --board_size={self.board_size} --embed_model=identity --value_model=random '
            '--policy_model=random --transition_model=random'.split())
        self.random_go_model = models.make_model(main.FLAGS)

    def test_new_trajectories(self):
        new_trajectories = game.new_traj_states(board_size=self.board_size, batch_size=2,
                                                trajectory_length=9)
        chex.assert_shape(new_trajectories, (2, 9, 6, 3, 3))
        np.testing.assert_array_equal(new_trajectories, jnp.zeros_like(new_trajectories))

    def test_random_sample_next_states_3x3_42rng(self):
        # We use the same RNG key that would be used in the update_trajectories function.
        actions, next_states = game.sample_actions_and_next_states(self.random_go_model, params={},
                                                                   rng_key=jax.random.fold_in(
                                                                       jax.random.PRNGKey(42), 0),
                                                                   states=gojax.new_states(
                                                                       board_size=self.board_size))
        np.testing.assert_array_equal(actions, [8])
        expected_next_states = gojax.decode_states("""
                                        _ _ _
                                        _ _ _
                                        _ _ B
                                        TURN=W
                                        """)
        np.testing.assert_array_equal(next_states, expected_next_states,
                                      gojax.get_string(next_states[0]))

    def test_update_trajectories_step_0(self):
        trajectories = game.Trajectories(
            nt_states=game.new_traj_states(board_size=self.board_size, batch_size=1,
                                           trajectory_length=6),
            nt_actions=jnp.full((1, self.board_size), fill_value=-1, dtype='uint16'))
        updated_data = game.update_trajectories(self.random_go_model, params={},
                                                rng_key=jax.random.PRNGKey(42), step=0,
                                                trajectories=trajectories)
        np.testing.assert_array_equal(updated_data.nt_states[:, 0],
                                      jnp.zeros_like(updated_data.nt_states[:, 0]))
        np.testing.assert_array_equal(updated_data.nt_actions[:, 0], [8])
        np.testing.assert_array_equal(updated_data.nt_states[:, 1], gojax.decode_states("""
                                                        _ _ _
                                                        _ _ _
                                                        _ _ B
                                                        TURN=W
                                                        """))

    def test_update_trajectories_step_1(self):
        trajectories = game.Trajectories(
            nt_states=game.new_traj_states(board_size=self.board_size, batch_size=1,
                                           trajectory_length=6),
            nt_actions=jnp.full((1, self.board_size), fill_value=-1, dtype='uint16'))
        updated_data = game.update_trajectories(self.random_go_model, params={},
                                                rng_key=jax.random.PRNGKey(42), step=1,
                                                trajectories=trajectories)
        np.testing.assert_array_equal(updated_data.nt_states[:, 0],
                                      jnp.zeros_like(updated_data.nt_states[:, 0]))
        np.testing.assert_array_equal(updated_data.nt_actions[:, 0],
                                      jnp.array([-1], dtype='uint16'))
        np.testing.assert_array_equal(updated_data.nt_states[:, 1],
                                      jnp.zeros_like(updated_data.nt_states[:, 1]))
        np.testing.assert_array_equal(updated_data.nt_actions[:, 1], [8])
        np.testing.assert_array_equal(updated_data.nt_states[:, 2], gojax.decode_states("""
                                                        _ _ _
                                                        _ _ _
                                                        _ _ B
                                                        TURN=W
                                                        """))

    def test_random_self_play_3x3_42rng(self):
        main.FLAGS('foo --batch_size=1 --board_size=3 --trajectory_length=3'.split())
        trajectories = game.self_play(main.FLAGS, self.random_go_model, params={},
                                      rng_key=jax.random.PRNGKey(42))
        expected_nt_states = gojax.decode_states("""
                                                    _ _ _
                                                    _ _ _
                                                    _ _ _
                                                    
                                                    _ _ _
                                                    _ _ _
                                                    _ _ B
                                                    TURN=W
                                                    
                                                    _ _ _
                                                    _ _ _
                                                    _ _ B
                                                    PASS=T
                                                    """)
        expected_nt_states = jnp.reshape(expected_nt_states, (1, 3, 6, 3, 3))

        def _get_nt_states_pretty_string(_nt_states, index=0):
            _pretty_traj_states_str = '\n'.join(
                map(lambda state: gojax.get_string(state), _nt_states[index]))
            return _pretty_traj_states_str

        pretty_traj_states_str = _get_nt_states_pretty_string(trajectories.nt_states)
        np.testing.assert_array_equal(trajectories.nt_states, expected_nt_states,
                                      pretty_traj_states_str)
        np.testing.assert_array_equal(trajectories.nt_actions,
                                      jnp.array([[8, 8, -1]], dtype='uint16'))

    def test_get_winners_one_tie_one_winning_one_winner(self):
        nt_states = game.new_traj_states(board_size=self.board_size, batch_size=3,
                                         trajectory_length=2)
        nt_states = nt_states.at[:1, 1].set(gojax.decode_states("""
                                                                    _ _ _
                                                                    _ _ _
                                                                    _ _ _
                                                                    """, ended=False))
        nt_states = nt_states.at[1:2, 1].set(gojax.decode_states("""
                                                                    _ _ _
                                                                    _ B _
                                                                    _ _ _
                                                                    """, ended=False))
        nt_states = nt_states.at[2:3, 1].set(gojax.decode_states("""
                                                                    _ _ _
                                                                    _ B _
                                                                    _ _ _
                                                                    """, ended=True))
        winners = game.get_winners(nt_states)
        np.testing.assert_array_equal(winners, [0, 1, 1])

    def test_get_labels_with_sample_trajectory(self):
        sample_trajectory = gojax.decode_states("""
                                                _ _ _
                                                _ _ _
                                                _ _ _
                                                
                                                _ _ _
                                                _ B _
                                                _ _ _
                                                TURN=W
                                                
                                                _ _ _
                                                _ B _
                                                _ _ _
                                                PASS=T
                                                """)
        sample_trajectory = jnp.reshape(sample_trajectory, (1, 3, 6, 3, 3))
        np.testing.assert_array_equal(game.get_labels(sample_trajectory), [[1, -1, 1]])

    def test_get_labels_with_komi(self):
        sample_trajectory = gojax.decode_states("""
                                            _ _ _
                                            _ _ _
                                            _ _ _

                                            _ _ _
                                            _ B _
                                            _ _ _
                                            TURN=W
                                            
                                            _ B _
                                            B W _
                                            _ B _
                                            
                                            _ B _
                                            B X B
                                            _ B _
                                            TURN=W
                                            """)
        np.testing.assert_array_equal(
            game.get_labels(jnp.reshape(sample_trajectory, (2, 2, 6, 3, 3))), [[1, -1], [1, -1]])

    def test_get_labels_multi_kill(self):
        sample_trajectory = gojax.decode_states("""
                                            B B _ 
                                            B W B 
                                            W _ W 
                                            PASS=T
                                            
                                            B B _ 
                                            B _ B 
                                            _ B _ 
                                            TURN=W
                                            """)
        np.testing.assert_array_equal(
            game.get_labels(jnp.reshape(sample_trajectory, (1, 2, 6, 3, 3))), [[1, -1]])


if __name__ == '__main__':
    unittest.main()
