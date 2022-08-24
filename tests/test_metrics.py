"""Tests metric functions."""
import tempfile
import unittest

import gojax
import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
import numpy as np
import pandas
from PIL import Image

from muzero_gojax import main
from muzero_gojax import metrics
from muzero_gojax import models


def test_plot_histogram_weights():
    """Tests histogram plot."""
    params = {
        'foo': {
            'w': jax.random.normal(jax.random.PRNGKey(1), (2, 2), dtype='bfloat16'),
            'b': jax.random.normal(jax.random.PRNGKey(2), (2, 2), dtype='bfloat16')
        }, 'bar': {'w': jax.random.normal(jax.random.PRNGKey(3), (2, 2), dtype='bfloat16')}
    }
    metrics.plot_histogram_weights(params)

    with tempfile.TemporaryFile() as file_pointer:
        plt.savefig(file_pointer)
        # Uncomment line below to update golden image.
        # plt.savefig('tests/test_data/histogram_weights_golden.png')
        file_pointer.seek(0)
        test_image = jnp.asarray(Image.open(file_pointer))
        expected_image = jnp.asarray(Image.open('tests/test_data/histogram_weights_golden.png'))
        diff_image = jnp.abs(test_image - expected_image)
        np.testing.assert_array_equal(diff_image, jnp.zeros_like(diff_image))


def test_plot_trajectories():
    """Tests trajectories plot."""
    trajectory_1 = jnp.expand_dims(gojax.decode_states("""
                        _ _ _
                        _ _ _
                        _ _ _

                        _ _ _
                        _ _ B
                        _ _ _
                        TURN=W"""), axis=0)
    trajectory_2 = jnp.expand_dims(gojax.decode_states("""
                        B _ _
                        W _ _
                        _ _ _
                        TURN=W

                        B _ _
                        W _ _
                        _ _ _
                        PASS=T"""), axis=0)
    trajectories = jnp.concatenate((trajectory_1, trajectory_2))
    metrics.plot_trajectories(trajectories)
    with tempfile.TemporaryFile() as file_pointer:
        plt.savefig(file_pointer)
        # Uncomment line below to update golden image.
        # plt.savefig('tests/test_data/trajectory_golden.png')
        file_pointer.seek(0)
        test_image = jnp.asarray(Image.open(file_pointer))
        expected_image = jnp.asarray(Image.open('tests/test_data/trajectory_golden.png'))
        diff_image = jnp.abs(test_image - expected_image)
        np.testing.assert_array_equal(diff_image, jnp.zeros_like(diff_image))


def test_plot_model_thoughts_with_interesting_states():
    """Tests model_thoughts plot."""
    main.FLAGS.unparse_flags()
    main.FLAGS('foo --board_size=3 --hdim=2 --embed_model=cnn_lite --value_model=linear '
               '--policy_model=linear --transition_model=cnn_lite'.split())
    go_model = models.make_model(main.FLAGS)
    states = metrics.get_interesting_states(board_size=3)
    params = go_model.init(jax.random.PRNGKey(42), states)
    metrics.plot_model_thoughts(go_model, params, states)

    with tempfile.TemporaryFile() as file_pointer:
        plt.savefig(file_pointer)
        # Uncomment line below to update golden image.
        # plt.savefig('tests/test_data/model_thoughts_golden.png')
        file_pointer.seek(0)
        test_image = jnp.asarray(Image.open(file_pointer))
        expected_image = jnp.asarray(Image.open('tests/test_data/model_thoughts_golden.png'))
        diff_image = jnp.abs(test_image - expected_image)
        np.testing.assert_array_equal(diff_image, jnp.zeros_like(diff_image))


def test_plot_metrics():
    """Tests metrics plot."""
    metrics_df = pandas.DataFrame({'foo': [0, 1, 2], 'bar': [-1, 1, -1]})
    metrics.plot_metrics(metrics_df)

    with tempfile.TemporaryFile() as file_pointer:
        plt.savefig(file_pointer)
        # Uncomment line below to update golden image.
        # plt.savefig('tests/test_data/metrics_golden.png')
        file_pointer.seek(0)
        test_image = jnp.asarray(Image.open(file_pointer))
        expected_image = jnp.asarray(Image.open('tests/test_data/metrics_golden.png'))
        diff_image = jnp.abs(test_image - expected_image)
        np.testing.assert_array_equal(diff_image, jnp.zeros_like(diff_image))


if __name__ == '__main__':
    unittest.main()
