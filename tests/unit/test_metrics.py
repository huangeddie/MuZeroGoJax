"""Tests metric functions."""
# pylint: disable=duplicate-code
import tempfile

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas
from absl.testing import absltest, flagsaver
from PIL import Image

from muzero_gojax import game, main, metrics, models

FLAGS = main.FLAGS


class MetricsTest(absltest.TestCase):
    """Tests metrics.py."""

    def setUp(self):
        FLAGS.mark_as_parsed()

    def test_plot_trajectories_on_random_trajectory_matches_golden_image(self):
        """Tests trajectories plot."""
        go_model = models.make_random_model()
        params = {}
        metrics.plot_sample_trajectories(
            game.new_trajectories(board_size=5,
                                  batch_size=8,
                                  trajectory_length=12), go_model, params)
        with tempfile.TemporaryFile() as file_pointer:
            plt.savefig(file_pointer)
            # Uncomment line below to update golden image.
            # plt.savefig('tests/unit/test_data/trajectory_golden.png')
            file_pointer.seek(0)
            test_image = jnp.asarray(Image.open(file_pointer))
            expected_image = jnp.asarray(
                Image.open('tests/unit/test_data/trajectory_golden.png'))
            diff_image = jnp.abs(test_image - expected_image)
            np.testing.assert_array_equal(diff_image,
                                          jnp.zeros_like(diff_image))

    @flagsaver.flagsaver(board_size=4,
                         hdim=2,
                         nlayers=0,
                         embed_model='non_spatial_conv',
                         value_model='non_spatial_conv',
                         policy_model='non_spatial_conv',
                         transition_model='non_spatial_conv')
    def test_plot_model_thoughts_on_interesting_states_matches_golden_image(
            self):
        """Tests model_thoughts plot."""
        go_model, params = models.build_model_with_params(
            FLAGS.board_size, FLAGS.dtype, jax.random.PRNGKey(FLAGS.rng))
        states = metrics.get_interesting_states(board_size=4)
        metrics.plot_model_thoughts(go_model, params, states)

        with tempfile.TemporaryFile() as file_pointer:
            plt.savefig(file_pointer)
            # Uncomment line below to update golden image.
            # plt.savefig('tests/unit/test_data/model_thoughts_golden.png')
            file_pointer.seek(0)
            test_image = jnp.asarray(Image.open(file_pointer))
            expected_image = jnp.asarray(
                Image.open('tests/unit/test_data/model_thoughts_golden.png'))
            diff_image = jnp.abs(test_image - expected_image)
            np.testing.assert_array_equal(diff_image,
                                          jnp.zeros_like(diff_image))

    def test_plot_metrics_matches_golden_image(self):
        """Tests metrics plot."""
        metrics_df = pandas.DataFrame({'foo': [0, 1, 2], 'bar': [-1, 1, -1]})
        metrics.plot_metrics(metrics_df)

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
