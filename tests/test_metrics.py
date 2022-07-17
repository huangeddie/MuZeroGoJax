import tempfile
import unittest

import gojax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from muzero_gojax import metrics


class MetricsTestCase(unittest.TestCase):
    def test_plot_trajectories(self):
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
        with tempfile.TemporaryFile() as fp:
            plt.savefig(fp)
            # Uncomment line below to update golden image.
            # plt.savefig('tests/test_data/trajectory_golden.png')
            fp.seek(0)
            test_image = jnp.asarray(Image.open(fp))
            expected_image = jnp.asarray(Image.open('tests/test_data/trajectory_golden.png'))
            diff_image = jnp.abs(test_image - expected_image)
            np.testing.assert_array_equal(diff_image, jnp.zeros_like(diff_image))


if __name__ == '__main__':
    unittest.main()
