import os.path
import tempfile
import unittest

import chex

from muzero_gojax import main

import jax.numpy as jnp

class MainCase(chex.TestCase):
    def test_maybe_save_model_no_save(self):
        main.FLAGS([''])
        params = {}
        self.assertIsNone(main.maybe_save_model(params, main.FLAGS))

    def test_maybe_save_model_saves_model(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            main.FLAGS(['', f'--save_dir={tmpdirname}'])
            params = {'foo': jnp.array(0, dtype='bfloat16')}
            filename = main.maybe_save_model(params, main.FLAGS)
            self.assertTrue(os.path.exists(filename))


if __name__ == '__main__':
    unittest.main()
