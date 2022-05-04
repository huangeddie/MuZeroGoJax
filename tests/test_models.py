import unittest

import gojax
import haiku as hk
import jax

import models


class ModelTestCase(unittest.TestCase):
    def test_get_random_model(self):
        model_fn = models.get_model('random')
        self.assertIsInstance(model_fn, hk.Transformed)
        params = model_fn.init(jax.random.PRNGKey(42), gojax.new_states(batch_size=2, board_size=3))
        self.assertIsInstance(params, dict)
        self.assertEqual(len(params), 0)

    def test_get_unknown_model(self):
        model_fn = models.get_model('foo')
        with self.assertRaises(KeyError):
            model_fn.init(jax.random.PRNGKey(42),
                          gojax.new_states(batch_size=2, board_size=3))


if __name__ == '__main__':
    unittest.main()
