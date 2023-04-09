"""Tests metric functions."""
# pylint: disable=duplicate-code
import tempfile

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas
from absl.testing import absltest, flagsaver
from PIL import Image

from muzero_gojax import data, game, main, metrics, models

FLAGS = main.FLAGS


class MetricsTest(absltest.TestCase):
    """Tests metrics.py."""

    def setUp(self):
        FLAGS.mark_as_parsed()

    def test_plot_trajectories_on_random_trajectory_matches_golden_image(self):
        """Tests trajectories plot."""
        go_model = models.make_random_policy_tromp_taylor_value_model()
        params = {}
        random_policy = models.get_policy_model(go_model, params)
        board_size = 5
        rng_key = jax.random.PRNGKey(42)
        random_traj: game.Trajectories = game.self_play(
            game.new_trajectories(board_size,
                                  batch_size=3,
                                  trajectory_length=2 * board_size**2),
            random_policy, rng_key)
        sampled_traj = data.sample_trajectories(random_traj,
                                                sample_size=8,
                                                rng_key=rng_key)
        metrics.plot_trajectories(sampled_traj,
                                  metrics.get_model_thoughts(
                                      go_model, params, sampled_traj, rng_key),
                                  title='Test Trajectories')
        with tempfile.TemporaryFile() as file_pointer:
            plt.savefig(file_pointer, dpi=50)
            # Uncomment line below to update golden image.
            plt.savefig('tests/unit/test_data/trajectory_golden.png', dpi=50)
            file_pointer.seek(0)
            test_image = jnp.asarray(Image.open(file_pointer))
            expected_image = jnp.asarray(
                Image.open('tests/unit/test_data/trajectory_golden.png'))
            diff_image = jnp.abs(test_image - expected_image)
            np.testing.assert_array_equal(diff_image,
                                          jnp.zeros_like(diff_image))

    def test_plot_train_metrics_matches_golden_image(self):
        """Tests metrics plot."""
        metrics_df = pandas.DataFrame({
            'avg_game_length':
            jax.random.normal(jax.random.PRNGKey(1), [3]),
            'black_win_pct':
            jax.random.normal(jax.random.PRNGKey(2), [3]),
            'tie_pct':
            jax.random.normal(jax.random.PRNGKey(3), [3]),
            'white_win_pct':
            jax.random.normal(jax.random.PRNGKey(4), [3]),
            'value_acc':
            jax.random.normal(jax.random.PRNGKey(5), [3]),
            'value_loss':
            jax.random.normal(jax.random.PRNGKey(6), [3]),
            'policy_entropy':
            jax.random.normal(jax.random.PRNGKey(7), [3]),
            'foo-winrate': [0, None, 1],
            'bar-winrate': [0, float('nan'), 1],
        })
        metrics.plot_train_metrics_by_regex(metrics_df)

        with tempfile.TemporaryFile() as file_pointer:
            plt.savefig(file_pointer, dpi=50)
            # Uncomment line below to update golden image.
            plt.savefig('tests/unit/test_data/metrics_golden.png', dpi=50)
            file_pointer.seek(0)
            test_image = jnp.asarray(Image.open(file_pointer))
            expected_image = jnp.asarray(
                Image.open('tests/unit/test_data/metrics_golden.png'))
            diff_image = jnp.abs(test_image - expected_image)
            np.testing.assert_array_equal(diff_image,
                                          jnp.zeros_like(diff_image))

    @flagsaver.flagsaver(training_steps=1,
                         eval_elo_frequency=1,
                         board_size=3,
                         batch_size=8)
    def test_eval_elo_with_multi_devices_noexcept(self):
        """Tests eval_elo with multi devices."""
        chex.assert_devices_available(8, 'CPU')
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        all_models_build_config = models.get_all_models_build_config(
            FLAGS.board_size, FLAGS.dtype)
        go_model, params = models.build_model_with_params(
            all_models_build_config, rng_key)
        metrics.eval_elo(go_model, params, FLAGS.board_size)


if __name__ == '__main__':
    absltest.main()
