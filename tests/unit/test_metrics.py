"""Tests metric functions."""
# pylint: disable=duplicate-code
import tempfile

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas
from absl.testing import absltest
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
            plt.savefig(file_pointer)
            # Uncomment line below to update golden image.
            plt.savefig('tests/unit/test_data/trajectory_golden.png')
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
            'black_wins':
            jax.random.normal(jax.random.PRNGKey(2), [3]),
            'ties':
            jax.random.normal(jax.random.PRNGKey(3), [3]),
            'white_wins':
            jax.random.normal(jax.random.PRNGKey(4), [3]),
            'value_acc':
            jax.random.normal(jax.random.PRNGKey(5), [3]),
            'value_loss':
            jax.random.normal(jax.random.PRNGKey(6), [3]),
            'policy_entropy':
            jax.random.normal(jax.random.PRNGKey(7), [3]),
            'foo-winrate':
            jax.random.normal(jax.random.PRNGKey(8), [3]),
        })
        metrics.plot_train_metrics_by_regex(metrics_df)

        with tempfile.TemporaryFile() as file_pointer:
            plt.savefig(file_pointer)
            # Uncomment line below to update golden image.
            # plt.savefig('tests/unit/test_data/metrics_golden.png')
            file_pointer.seek(0)
            test_image = jnp.asarray(Image.open(file_pointer))
            expected_image = jnp.asarray(
                Image.open('tests/unit/test_data/metrics_golden.png'))
            diff_image = jnp.abs(test_image - expected_image)
            np.testing.assert_array_equal(diff_image,
                                          jnp.zeros_like(diff_image))


if __name__ == '__main__':
    absltest.main()
