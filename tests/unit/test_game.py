"""Tests game.py."""
# pylint: disable=missing-function-docstring,duplicate-code,too-many-public-methods,unnecessary-lambda
import unittest

import chex
import gojax
import jax.numpy as jnp
import jax.random
import numpy as np
from absl.testing import flagsaver

from muzero_gojax import data, game, main, models, nt_utils

FLAGS = main.FLAGS


class GameTestCase(chex.TestCase):
    """Tests game.py."""

    def setUp(self):
        self.board_size = 3
        FLAGS(
            f'foo --board_size={self.board_size} --embed_model=non_spatial_conv '
            '--value_model=non_spatial_conv --policy_model=non_spatial_conv '
            '--transition_model=non_spatial_conv'.split())
        self.linear_go_model, self.params = models.build_model_with_params(
            FLAGS.board_size, FLAGS.dtype, jax.random.PRNGKey(FLAGS.rng))

    def test_new_trajectories_shape(self):
        new_trajectories = game.new_trajectories(board_size=self.board_size,
                                                 batch_size=2,
                                                 trajectory_length=9)

        chex.assert_shape(new_trajectories.nt_states, (2, 9, 6, 3, 3))
        chex.assert_shape(new_trajectories.nt_actions, (2, 9))

    def test_new_trajectories_has_zero_like_states(self):
        new_trajectories = game.new_trajectories(board_size=self.board_size,
                                                 batch_size=2,
                                                 trajectory_length=9)

        np.testing.assert_array_equal(
            new_trajectories.nt_states,
            jnp.zeros_like(new_trajectories.nt_states))

    def test_new_trajectories_initial_actions_are_max_value(self):
        new_trajectories = game.new_trajectories(board_size=self.board_size,
                                                 batch_size=2,
                                                 trajectory_length=9)

        np.testing.assert_array_equal(
            new_trajectories.nt_actions,
            jnp.full_like(new_trajectories.nt_actions,
                          fill_value=-1,
                          dtype='uint16'))

    def test_random_self_play_3x3_42rng_matches_golden_trajectory(self):
        trajectories = game.self_play(game.new_trajectories(
            batch_size=1, board_size=3, trajectory_length=3),
                                      self.linear_go_model,
                                      self.params,
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

        pretty_traj_states_str = _get_nt_states_pretty_string(
            trajectories.nt_states)
        np.testing.assert_array_equal(trajectories.nt_states,
                                      expected_nt_states,
                                      pretty_traj_states_str)
        np.testing.assert_array_equal(trajectories.nt_actions,
                                      jnp.array([[8, 8, -1]], dtype='uint16'))

    def test_random_5x5_self_play_yields_black_advantage(self):
        trajectories = game.self_play(game.new_trajectories(
            batch_size=128, board_size=5, trajectory_length=24),
                                      models.make_random_model(),
                                      params={},
                                      rng_key=jax.random.PRNGKey(42))

        game_winners = game.get_nt_player_labels(trajectories.nt_states)

        black_winrate = jnp.mean(game_winners[:, ::2] == 1, dtype='bfloat16')
        white_winrate = jnp.mean(game_winners[:, 1::2] == 1, dtype='bfloat16')
        tie_rate = jnp.mean(game_winners == 0, dtype='bfloat16')

        self.assertBetween(black_winrate, 0.45, 0.55)
        self.assertBetween(white_winrate, 0.25, 0.35)
        self.assertBetween(tie_rate, 0.2, 0.5)

    def test_get_labels_on_single_trajectory(self):
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
        np.testing.assert_array_equal(
            game.get_nt_player_labels(sample_trajectory), [[1, -1, 1]])

    def test_get_labels_on_states_with_komi(self):
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
            game.get_nt_player_labels(
                jnp.reshape(sample_trajectory, (2, 2, 6, 3, 3))),
            [[1, -1], [1, -1]])

    def test_get_labels_on_states_with_multi_kill(self):
        sample_nt_states = gojax.decode_states("""
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
            game.get_nt_player_labels(
                jnp.reshape(sample_nt_states, (1, 2, 6, 3, 3))), [[1, -1]])

    def test_rotationally_augments_four_equal_single_length_trajectories_on_3x3_board(
            self):
        states = gojax.decode_states("""
                                        B _ _
                                        _ _ _
                                        _ _ _
                                        
                                        B _ _
                                        _ _ _
                                        _ _ _
                                        
                                        B _ _
                                        _ _ _
                                        _ _ _
                                        
                                        B _ _
                                        _ _ _
                                        _ _ _
                                        """)
        nt_states = nt_utils.unflatten_first_dim(states, 4, 1)

        expected_rot_aug_states = gojax.decode_states("""
                                                    B _ _
                                                    _ _ _
                                                    _ _ _
                
                                                    _ _ _
                                                    _ _ _
                                                    B _ _
                
                                                    _ _ _
                                                    _ _ _
                                                    _ _ B
                
                                                    _ _ B
                                                    _ _ _
                                                    _ _ _
                                                    """)
        expected_rot_aug_nt_states = nt_utils.unflatten_first_dim(
            expected_rot_aug_states, 4, 1)
        filler_nt_actions = jnp.zeros((4, 1), dtype='uint16')
        rot_traj = game.rotationally_augment_trajectories(
            data.Trajectories(nt_states=nt_states,
                              nt_actions=filler_nt_actions))
        np.testing.assert_array_equal(rot_traj.nt_states,
                                      expected_rot_aug_nt_states)

    def test_rotationally_augments_start_states_are_noops(self):
        states = gojax.new_states(board_size=3, batch_size=4)
        nt_states = nt_utils.unflatten_first_dim(states, 4, 1)

        filler_nt_actions = jnp.zeros((4, 1), dtype='uint16')
        rot_traj = game.rotationally_augment_trajectories(
            data.Trajectories(nt_states=nt_states,
                              nt_actions=filler_nt_actions))
        np.testing.assert_array_equal(rot_traj.nt_states, nt_states)

    def test_rotationally_augment_pass_actions_are_noops(self):
        indicator_actions = jnp.repeat(jnp.array([[[0, 0, 0], [0, 0, 0],
                                                   [0, 0, 0]]]),
                                       axis=0,
                                       repeats=4)
        expected_indicator_actions = jnp.array([[[0, 0, 0], [0, 0, 0],
                                                 [0, 0, 0]],
                                                [[0, 0, 0], [0, 0, 0],
                                                 [0, 0, 0]],
                                                [[0, 0, 0], [0, 0, 0],
                                                 [0, 0, 0]],
                                                [[0, 0, 0], [0, 0, 0],
                                                 [0, 0, 0]]])

        nt_actions = nt_utils.unflatten_first_dim(
            gojax.action_indicator_to_1d(indicator_actions), 4, 1)
        expected_nt_actions = nt_utils.unflatten_first_dim(
            gojax.action_indicator_to_1d(expected_indicator_actions), 4, 1)

        filler_nt_states = nt_utils.unflatten_first_dim(
            gojax.new_states(board_size=3, batch_size=4), 4, 1)
        rot_traj = game.rotationally_augment_trajectories(
            data.Trajectories(nt_states=filler_nt_states,
                              nt_actions=nt_actions))
        np.testing.assert_array_equal(rot_traj.nt_actions, expected_nt_actions)

    def test_rotationally_augments_states_on_4x1_trajectory_with_3x3_board(
            self):
        states = gojax.decode_states("""
                                    B _ _
                                    _ _ _
                                    _ _ _
                                    """)
        nt_states = jnp.repeat(nt_utils.unflatten_first_dim(states, 1, 1),
                               axis=0,
                               repeats=4)

        expected_rot_aug_states = gojax.decode_states("""
                                                    B _ _
                                                    _ _ _
                                                    _ _ _

                                                    _ _ _
                                                    _ _ _
                                                    B _ _

                                                    _ _ _
                                                    _ _ _
                                                    _ _ B

                                                    _ _ B
                                                    _ _ _
                                                    _ _ _
                                                    """)
        expected_rot_aug_nt_states = nt_utils.unflatten_first_dim(
            expected_rot_aug_states, 4, 1)
        filler_nt_actions = jnp.zeros((4, 1), dtype='uint16')
        rot_traj = game.rotationally_augment_trajectories(
            data.Trajectories(nt_states=nt_states,
                              nt_actions=filler_nt_actions))
        np.testing.assert_array_equal(rot_traj.nt_states,
                                      expected_rot_aug_nt_states)

    def test_rotationally_augments_actions_on_4x1_trajectory_with_3x3_board(
            self):
        nt_actions = jnp.zeros((4, 1), dtype='uint16')
        expected_nt_actions = jnp.array([[0], [6], [8], [2]], dtype='uint16')
        filler_nt_states = nt_utils.unflatten_first_dim(
            gojax.new_states(board_size=3, batch_size=4), 4, 1)

        rot_traj = game.rotationally_augment_trajectories(
            data.Trajectories(nt_states=filler_nt_states,
                              nt_actions=nt_actions))

        np.testing.assert_array_equal(rot_traj.nt_actions, expected_nt_actions)

    def test_rotationally_augments_states_on_8x1_trajectory_with_3x3_board(
            self):
        states = gojax.decode_states("""
                                    B _ _
                                    _ _ _
                                    _ _ _
                                    """)
        nt_states = jnp.repeat(nt_utils.unflatten_first_dim(states, 1, 1),
                               axis=0,
                               repeats=8)

        expected_rot_aug_states = gojax.decode_states("""
                                                    B _ _
                                                    _ _ _
                                                    _ _ _
                                                    
                                                    B _ _
                                                    _ _ _
                                                    _ _ _

                                                    _ _ _
                                                    _ _ _
                                                    B _ _
                                                    
                                                    _ _ _
                                                    _ _ _
                                                    B _ _

                                                    _ _ _
                                                    _ _ _
                                                    _ _ B
                                                    
                                                    _ _ _
                                                    _ _ _
                                                    _ _ B

                                                    _ _ B
                                                    _ _ _
                                                    _ _ _
                                                    
                                                    _ _ B
                                                    _ _ _
                                                    _ _ _
                                                    """)
        expected_rot_aug_nt_states = nt_utils.unflatten_first_dim(
            expected_rot_aug_states, 8, 1)
        filler_nt_actions = jnp.zeros((8, 1), dtype='uint16')
        rot_traj = game.rotationally_augment_trajectories(
            data.Trajectories(nt_states=nt_states,
                              nt_actions=filler_nt_actions))
        np.testing.assert_array_equal(rot_traj.nt_states,
                                      expected_rot_aug_nt_states)

    def test_rotationally_augments_actions_on_8x1_trajectory_with_3x3_board(
            self):
        nt_actions = jnp.zeros((8, 1), dtype='uint16')
        expected_nt_actions = jnp.array(
            [[0], [0], [6], [6], [8], [8], [2], [2]], dtype='uint16')
        filler_nt_states = nt_utils.unflatten_first_dim(
            gojax.new_states(board_size=3, batch_size=8), 8, 1)

        rot_traj = game.rotationally_augment_trajectories(
            data.Trajectories(nt_states=filler_nt_states,
                              nt_actions=nt_actions))

        np.testing.assert_array_equal(rot_traj.nt_actions, expected_nt_actions)

    def test_rot_augments_states_consistently_in_same_traj_on_2x2_traj_with_3x3_board(
            self):
        states = gojax.decode_states("""
                                    B _ _
                                    _ _ _
                                    _ _ _
                                    
                                    B W _
                                    _ _ _
                                    _ _ _
                                    
                                    B _ _
                                    _ _ _
                                    _ _ _
                                    
                                    B W _
                                    _ _ _
                                    _ _ _
                                    """)

        nt_states = nt_utils.unflatten_first_dim(states, 2, 2)

        expected_rot_aug_states = gojax.decode_states("""
                                                    B _ _
                                                    _ _ _
                                                    _ _ _
                                                    
                                                    B W _
                                                    _ _ _
                                                    _ _ _
                                                    
                                                    _ _ _
                                                    _ _ _
                                                    B _ _

                                                    _ _ _
                                                    W _ _
                                                    B _ _
                                                    """)
        expected_rot_aug_nt_states = nt_utils.unflatten_first_dim(
            expected_rot_aug_states, 2, 2)
        filler_nt_actions = jnp.zeros((2, 2), dtype='uint16')
        rot_traj = game.rotationally_augment_trajectories(
            data.Trajectories(nt_states=nt_states,
                              nt_actions=filler_nt_actions))
        np.testing.assert_array_equal(rot_traj.nt_states,
                                      expected_rot_aug_nt_states)

    def test_rot_augments_actions_consistently_in_same_traj_on_2x2_traj_with_3x3_board(
            self):
        nt_actions = jnp.zeros((2, 2), dtype='uint16')
        expected_nt_actions = jnp.array([[0, 0], [6, 6]], dtype='uint16')

        filler_nt_states = nt_utils.unflatten_first_dim(
            gojax.new_states(board_size=3, batch_size=4), 2, 2)
        rot_traj = game.rotationally_augment_trajectories(
            data.Trajectories(nt_states=filler_nt_states,
                              nt_actions=nt_actions))
        np.testing.assert_array_equal(rot_traj.nt_actions, expected_nt_actions)

    def test_random_models_are_equally_matched(self):
        with flagsaver.flagsaver(rng=42,
                                 embed_model='identity',
                                 value_model='random',
                                 policy_model='random',
                                 transition_model='random',
                                 decode_model='amplified',
                                 board_size=5):
            random_model_a, a_params = models.build_model_with_params(
                FLAGS.board_size, FLAGS.dtype, jax.random.PRNGKey(FLAGS.rng))
            a_policy = models.get_policy_model(random_model_a, a_params)

        with flagsaver.flagsaver(rng=69,
                                 embed_model='identity',
                                 value_model='random',
                                 policy_model='random',
                                 transition_model='random',
                                 decode_model='amplified',
                                 board_size=5):
            random_model_b, b_params = models.build_model_with_params(
                FLAGS.board_size, FLAGS.dtype, jax.random.PRNGKey(FLAGS.rng))
            b_policy = models.get_policy_model(random_model_b, b_params)

        win_a, _, win_b = game.pit(a_policy,
                                   b_policy,
                                   FLAGS.board_size,
                                   n_games=128,
                                   traj_len=26)
        self.assertAlmostEqual(win_a, win_b)

    def test_tromp_taylor_has_85_pct_winrate_against_random(self):
        with flagsaver.flagsaver(rng=42,
                                 embed_model='identity',
                                 value_model='random',
                                 policy_model='random',
                                 transition_model='random',
                                 decode_model='amplified',
                                 board_size=5):
            random_model, random_params = models.build_model_with_params(
                FLAGS.board_size, FLAGS.dtype, jax.random.PRNGKey(FLAGS.rng))
            random_policy = models.get_policy_model(random_model,
                                                    random_params)

        with flagsaver.flagsaver(rng=69,
                                 embed_model='identity',
                                 value_model='random',
                                 policy_model='tromp_taylor',
                                 transition_model='random',
                                 decode_model='amplified',
                                 board_size=5):
            tromp_taylor_model, tromp_taylor_params = models.build_model_with_params(
                FLAGS.board_size, FLAGS.dtype, jax.random.PRNGKey(FLAGS.rng))
            tromp_taylor_policy = models.get_policy_model(
                tromp_taylor_model, tromp_taylor_params)

        win_a, _, _ = game.pit(tromp_taylor_policy,
                               random_policy,
                               FLAGS.board_size,
                               n_games=128,
                               traj_len=26)
        self.assertAlmostEqual(win_a / 128, 0.85, delta=0.01)

    def test_random_has_10_pct_winrate_against_tromp_taylor(self):
        with flagsaver.flagsaver(rng=42,
                                 embed_model='identity',
                                 value_model='random',
                                 policy_model='random',
                                 transition_model='random',
                                 decode_model='amplified',
                                 board_size=5):
            random_model, random_params = models.build_model_with_params(
                FLAGS.board_size, FLAGS.dtype, jax.random.PRNGKey(FLAGS.rng))
            random_policy = models.get_policy_model(random_model,
                                                    random_params)

        with flagsaver.flagsaver(rng=69,
                                 embed_model='identity',
                                 value_model='random',
                                 policy_model='tromp_taylor',
                                 transition_model='random',
                                 decode_model='amplified',
                                 board_size=5):
            tromp_taylor_model, tromp_taylor_params = models.build_model_with_params(
                FLAGS.board_size, FLAGS.dtype, jax.random.PRNGKey(FLAGS.rng))
            tromp_taylor_policy = models.get_policy_model(
                tromp_taylor_model, tromp_taylor_params)

        win_a, _, _ = game.pit(random_policy,
                               tromp_taylor_policy,
                               FLAGS.board_size,
                               n_games=128,
                               traj_len=26)
        self.assertAlmostEqual(win_a / 128, 0.10, delta=0.01)


if __name__ == '__main__':
    unittest.main()
