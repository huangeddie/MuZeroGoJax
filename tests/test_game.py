"""Tests game.py."""
# pylint: disable=missing-function-docstring,no-self-use,unnecessary-lambda,
# pylint: disable=no-value-for-parameter,duplicate-code

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
        main.FLAGS(f'foo --board_size={self.board_size} --embed_model=identity --value_model=random '
                   '--policy_model=random --transition_model=random'.split())
        self.random_go_model = models.make_model(main.FLAGS)

    def test_new_trajectories(self):
        new_trajectories = game.new_trajectories(board_size=self.board_size, batch_size=2, max_num_steps=9)
        chex.assert_shape(new_trajectories, (2, 9, 6, 3, 3))
        np.testing.assert_array_equal(new_trajectories, jnp.zeros_like(new_trajectories))

    def test_random_sample_next_states_3x3_42rng(self):
        # We use the same RNG key that would be used in the update_trajectories function.
        next_states = game.sample_next_states(self.random_go_model, params={}, model_state={},
                                              rng_key=jax.random.fold_in(jax.random.PRNGKey(42), 0),
                                              states=gojax.new_states(board_size=self.board_size))
        expected_next_states = gojax.decode_states("""
                                        _ _ _
                                        _ _ _
                                        _ _ B
                                        """, turn=gojax.WHITES_TURN)
        np.testing.assert_array_equal(next_states, expected_next_states)

    def test_update_trajectories_step_0(self):
        trajectories = game.new_trajectories(board_size=self.board_size, batch_size=1, max_num_steps=6)
        updated_trajectories = game.update_trajectories(self.random_go_model, params={}, model_state={},
                                                        rng_key=jax.random.PRNGKey(42), step=0,
                                                        trajectories=trajectories)
        np.testing.assert_array_equal(updated_trajectories[:, 0], jnp.zeros_like(updated_trajectories[:, 0]))
        np.testing.assert_array_equal(updated_trajectories[:, 1], gojax.decode_states("""
                                                        _ _ _
                                                        _ _ _
                                                        _ _ B
                                                        """, turn=gojax.WHITES_TURN))

    def test_update_trajectories_step_1(self):
        trajectories = game.new_trajectories(board_size=self.board_size, batch_size=1, max_num_steps=6)
        updated_trajectories = game.update_trajectories(self.random_go_model, params={}, model_state={},
                                                        rng_key=jax.random.PRNGKey(42), step=1,
                                                        trajectories=trajectories)
        np.testing.assert_array_equal(updated_trajectories[:, 0], jnp.zeros_like(updated_trajectories[:, 0]))
        np.testing.assert_array_equal(updated_trajectories[:, 1], jnp.zeros_like(updated_trajectories[:, 1]))
        np.testing.assert_array_equal(updated_trajectories[:, 2], gojax.decode_states("""
                                                        _ _ _
                                                        _ _ _
                                                        _ _ B
                                                        """, turn=gojax.WHITES_TURN))

    def test_random_self_play_3x3_42rng(self):
        main.FLAGS('foo --batch_size=1 --board_size=3 --max_num_steps=6'.split())
        trajectories = game.self_play(main.FLAGS, self.random_go_model, params={}, model_state={},
                                      rng_key=jax.random.PRNGKey(42))
        expected_trajectories = gojax.decode_states("""
                                                    _ _ _
                                                    _ _ _
                                                    _ _ _
                                                    
                                                    _ _ _
                                                    _ _ _
                                                    _ _ B
                                                    TURN=W
                                                    
                                                    _ _ _
                                                    _ _ _
                                                    _ W B
                                                    
                                                    _ B _
                                                    _ _ _
                                                    _ W B
                                                    TURN=W
                                                    
                                                    _ B _
                                                    _ W _
                                                    _ W B
                                                    
                                                    _ B B
                                                    _ W _
                                                    _ W B
                                                    TURN=W
                                                    """)
        expected_trajectories = jnp.reshape(expected_trajectories, (1, 6, 6, 3, 3))

        def _get_trajectory_pretty_string(trajectories, index=0):
            pretty_trajectory_str = '\n'.join(map(lambda state: gojax.get_string(state), trajectories[index]))
            return pretty_trajectory_str

        pretty_trajectory_str = _get_trajectory_pretty_string(trajectories)
        np.testing.assert_array_equal(trajectories, expected_trajectories, pretty_trajectory_str)

    def test_get_winners_one_tie_one_winning_one_winner(self):
        trajectories = game.new_trajectories(board_size=self.board_size, batch_size=3, max_num_steps=2)
        trajectories = trajectories.at[:1, 1].set(gojax.decode_states("""
                                                                    _ _ _
                                                                    _ _ _
                                                                    _ _ _
                                                                    """, ended=False))
        trajectories = trajectories.at[1:2, 1].set(gojax.decode_states("""
                                                                    _ _ _
                                                                    _ B _
                                                                    _ _ _
                                                                    """, ended=False))
        trajectories = trajectories.at[2:3, 1].set(gojax.decode_states("""
                                                                    _ _ _
                                                                    _ B _
                                                                    _ _ _
                                                                    """, ended=True))
        winners = game.get_winners(trajectories)
        np.testing.assert_array_equal(winners, [0, 1, 1])

    def test_get_actions_and_labels_with_sample_trajectory(self):
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
        actions, labels = game.get_actions_and_labels(sample_trajectory)
        np.testing.assert_array_equal(actions, [[4, 9, 9]])
        np.testing.assert_array_equal(labels, [[1, -1, 1]])

    def test_get_actions_and_labels_with_komi(self):
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
        actions, labels = game.get_actions_and_labels(jnp.reshape(sample_trajectory, (2, 2, 6, 3, 3)))
        np.testing.assert_array_equal(actions, [[4, 1], [5, 9]])
        np.testing.assert_array_equal(labels, [[1, -1], [1, -1]])

    def test_get_actions_and_labels_multi_kill(self):
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
        actions, labels = game.get_actions_and_labels(jnp.reshape(sample_trajectory, (1, 2, 6, 3, 3)))
        np.testing.assert_array_equal(actions, [[7, 4]])
        np.testing.assert_array_equal(labels, [[1, -1]])




if __name__ == '__main__':
    unittest.main()
