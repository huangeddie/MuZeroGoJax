"""Tests manager module."""
# pylint: disable=too-many-public-methods,missing-function-docstring
import os
import tempfile
import unittest

import chex
import jax
from absl.testing import flagsaver

from muzero_gojax import main, manager, models

FLAGS = main.FLAGS


class ManagerCase(chex.TestCase):
    """Tests manager module."""

    def setUp(self):
        FLAGS.mark_as_parsed()

    # We need > 5 train steps because the first _TRAJECTORY_BUFFER_SIZE=4 steps are 0
    # to fill up the buffer, and the warmup scheduling causes the next learning rate to
    # be 0.
    @flagsaver.flagsaver(training_steps=6, board_size=3)
    def test_train_model_changes_params(self):
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        model_build_config = models.get_model_build_config(FLAGS.board_size)
        go_model, params = models.build_model_with_params(
            model_build_config, rng_key)
        # We must copy the params because train_model donates them.
        original_params = jax.tree_map(lambda x: x.copy(), params)
        new_params, _ = manager.train_model(go_model, params,
                                            model_build_config, rng_key)
        with self.assertRaises(AssertionError):
            chex.assert_trees_all_equal(original_params, new_params)

    @flagsaver.flagsaver(training_steps=1, board_size=3)
    def test_train_model_metrics_df_matches_golden_format(self):
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        model_build_config = models.get_model_build_config(FLAGS.board_size)
        go_model, params = models.build_model_with_params(
            model_build_config, rng_key)
        _, metrics_df = manager.train_model(go_model, params,
                                            model_build_config, rng_key)
        self.assertEqual(metrics_df.index.name, 'step')
        self.assertEqual(
            set(metrics_df.columns), {
                'value_loss',
                'policy_loss',
                'white_win_pct',
                'hypo_value_acc',
                'area_acc',
                'hypo_area_acc',
                'policy_acc',
                'pass_rate',
                'policy_entropy',
                'partial_qval_entropy',
                'hypo_area_loss',
                'piece_collision_rate',
                'tie_pct',
                'hypo_value_loss',
                'black_win_pct',
                'avg_game_length',
                'area_loss',
                'value_acc',
            })

    @flagsaver.flagsaver(training_steps=1,
                         board_size=3,
                         batch_size=512,
                         area_model='RandomArea')
    def test_train_model_area_acc_roughly_50_pct(self):
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        model_build_config = models.get_model_build_config(FLAGS.board_size)
        go_model, params = models.build_model_with_params(
            model_build_config, rng_key)
        _, metrics_df = manager.train_model(go_model, params,
                                            model_build_config, rng_key)
        self.assertAlmostEqual(metrics_df['area_acc'].mean(), 0.5, places=1)

    @flagsaver.flagsaver(training_steps=1, board_size=3, eval_elo_frequency=1)
    def test_train_model_with_eval_metrics_df_matches_golden_format(self):
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        model_build_config = models.get_model_build_config(FLAGS.board_size)
        go_model, params = models.build_model_with_params(
            model_build_config, rng_key)
        _, metrics_df = manager.train_model(go_model, params,
                                            model_build_config, rng_key)
        self.assertEqual(metrics_df.index.name, 'step')
        self.assertEqual(
            set(metrics_df.columns), {
                'value_loss',
                'policy_loss',
                'white_win_pct',
                'hypo_value_acc',
                'area_acc',
                'hypo_area_acc',
                'policy_acc',
                'pass_rate',
                'policy_entropy',
                'partial_qval_entropy',
                'hypo_area_loss',
                'piece_collision_rate',
                'tie_pct',
                'hypo_value_loss',
                'black_win_pct',
                'avg_game_length',
                'area_loss',
                'value_acc',
                'Tromp Taylor Amplified-winrate',
            })

    def test_multi_update_steps_params_differ_from_single_update_step_params(
            self):
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        model_build_config = models.get_model_build_config(FLAGS.board_size)
        go_model, params = models.build_model_with_params(
            model_build_config, rng_key)
        # We must copy the params because train_model donates them.
        with flagsaver.flagsaver(training_steps=2, board_size=3):
            single_update_params, _ = manager.train_model(
                go_model, jax.tree_map(lambda x: x.copy(), params),
                model_build_config, rng_key)
        with flagsaver.flagsaver(training_steps=2,
                                 board_size=3,
                                 model_updates_per_train_step=2):
            two_update_params, _ = manager.train_model(
                go_model, jax.tree_map(lambda x: x.copy(), params),
                model_build_config, rng_key)
        with self.assertRaises(AssertionError):
            chex.assert_trees_all_equal(single_update_params,
                                        two_update_params)

    @flagsaver.flagsaver(training_steps=6,
                         board_size=3,
                         log_training_frequency=2)
    def test_train_model_sparse_eval_changes_params(self):
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        model_build_config = models.get_model_build_config(FLAGS.board_size)
        go_model, params = models.build_model_with_params(
            model_build_config, rng_key)
        # We must copy the params because train_model donates them.
        new_params, _ = manager.train_model(
            go_model, jax.tree_map(lambda x: x.copy(), params),
            model_build_config, rng_key)
        with self.assertRaises(AssertionError):
            chex.assert_trees_all_equal(params, new_params)

    @flagsaver.flagsaver(training_steps=1,
                         board_size=3,
                         self_play_model='random')
    def test_train_model_with_random_self_play_noexcept(self):
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        model_build_config = models.get_model_build_config(FLAGS.board_size)
        go_model, params = models.build_model_with_params(
            model_build_config, rng_key)
        manager.train_model(go_model, params, model_build_config, rng_key)

    @flagsaver.flagsaver(training_steps=2,
                         board_size=3,
                         batch_size=16,
                         pmap=True)
    def test_train_model_with_pmap_returns_params_on_first_device(self):
        chex.assert_devices_available(8, 'CPU')
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        model_build_config = models.get_model_build_config(FLAGS.board_size)
        go_model, params = models.build_model_with_params(
            model_build_config, rng_key)
        params, _ = manager.train_model(go_model, params, model_build_config,
                                        rng_key)
        chex.assert_tree_is_on_device(params, device=jax.devices()[0])

    @flagsaver.flagsaver(training_steps=1,
                         eval_elo_frequency=1,
                         board_size=3,
                         batch_size=8,
                         pmap=True)
    def test_train_model_with_pmap_evals_elo_noexcept(self):
        chex.assert_devices_available(8, 'CPU')
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        model_build_config = models.get_model_build_config(FLAGS.board_size)
        go_model, params = models.build_model_with_params(
            model_build_config, rng_key)
        params, _ = manager.train_model(go_model, params, model_build_config,
                                        rng_key)

    @flagsaver.flagsaver(training_steps=1,
                         save_model_frequency=1,
                         board_size=3)
    def test_train_model_saves_model(self):
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        model_build_config = models.get_model_build_config(FLAGS.board_size)
        go_model, params = models.build_model_with_params(
            model_build_config, rng_key)
        with tempfile.TemporaryDirectory() as model_dir:
            params, _ = manager.train_model(go_model,
                                            params,
                                            model_build_config,
                                            rng_key,
                                            save_dir=model_dir)
            self.assertTrue(
                os.path.exists(os.path.join(model_dir, 'params.npz')))

    @flagsaver.flagsaver(training_steps=1,
                         save_model_frequency=1,
                         board_size=3)
    def test_train_model_gracefully_handles_save_error(self):
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        model_build_config = models.get_model_build_config(FLAGS.board_size)
        go_model, params = models.build_model_with_params(
            model_build_config, rng_key)
        with tempfile.TemporaryDirectory() as tmpdirname:
            bad_model_dir = os.path.join(tmpdirname, 'foo')
            params, _ = manager.train_model(go_model,
                                            params,
                                            model_build_config,
                                            rng_key,
                                            save_dir=bad_model_dir)
            self.assertFalse(os.path.exists(bad_model_dir))


if __name__ == '__main__':
    unittest.main()
