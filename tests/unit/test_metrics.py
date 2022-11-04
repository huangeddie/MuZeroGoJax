"""Tests metric functions."""
# pylint: disable=duplicate-code
import tempfile

import gojax
import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
import numpy as np
import pandas
from PIL import Image
from absl.testing import absltest
from absl.testing import flagsaver

from muzero_gojax import game
from muzero_gojax import main
from muzero_gojax import metrics
from muzero_gojax import models
from muzero_gojax import data

FLAGS = main.FLAGS


class MetricsTest(absltest.TestCase):
    """Tests metrics.py."""

    def setUp(self):
        FLAGS.mark_as_parsed()

    def test_plot_trajectories_matches_golden_image(self):
        """Tests trajectories plot."""
        trajectories = data.Trajectories(nt_states=jnp.reshape(
            gojax.decode_states("""
                            _ _ _
                            _ _ _
                            _ _ _
    
                            _ _ _
                            _ _ B
                            _ _ _
                            TURN=W
                            
                            B _ _
                            W _ _
                            _ _ _
                            TURN=W
    
                            B _ _
                            W _ _
                            _ _ _
                            PASS=T
                            """), (2, 2, 6, 3, 3)),
                                         nt_actions=jnp.array(
                                             [[5, -1], [9, -1]],
                                             dtype='uint16'))
        rng_key = jax.random.PRNGKey(42)
        metrics.plot_trajectories(
            trajectories,
            nt_policy_logits=jax.random.normal(rng_key, (2, 2, 10)),
            nt_value_logits=jax.random.normal(rng_key, (2, 2)))
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

    @flagsaver.flagsaver(board_size=4,
                         hdim=2,
                         embed_model='linear_conv',
                         value_model='linear_conv',
                         policy_model='linear_conv',
                         transition_model='linear_conv')
    def test_plot_model_thoughts_on_interesting_states_matches_golden_image(
            self):
        """Tests model_thoughts plot."""
        go_model, params = models.build_model(FLAGS.board_size, FLAGS.dtype)
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
