import os.path
import pickle
import tempfile
import unittest

import chex
import haiku as hk
import jax.numpy as jnp
import jax.random
import numpy as np

from muzero_gojax import main
from muzero_gojax import models


class MainCase(chex.TestCase):
    def setUp(self):
        main.FLAGS.unparse_flags()

    def test_maybe_save_model_no_save(self):
        main.FLAGS([''])
        params = {}
        self.assertIsNone(main.maybe_save_model(params, main.FLAGS))

    def test_maybe_save_model_saves_model_with_bfloat16_type(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            main.FLAGS(['', f'--save_dir={tmpdirname}'])
            params = {'foo': jnp.array(0, dtype='bfloat16')}
            filename = main.maybe_save_model(params, main.FLAGS)
            self.assertTrue(os.path.exists(filename))

    def test_load_model(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            main.FLAGS(['', f'--save_dir={tmpdirname}'])
            model = hk.transform(
                lambda x: models.value.Linear3DValue(main.FLAGS.board_size, hdim=None)(x))
            rng_key = jax.random.PRNGKey(main.FLAGS.random_seed)
            x = jax.random.normal(rng_key, (1, 6, 5, 5))
            params = model.init(rng_key, x)
            expected_output = model.apply(params, rng_key, x)
            filename = main.maybe_save_model(params, main.FLAGS)
            with open(filename, 'rb') as f:
                params = pickle.load(f)
            np.testing.assert_array_equal(model.apply(params, rng_key, x), expected_output)


if __name__ == '__main__':
    unittest.main()
