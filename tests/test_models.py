"""Tests model.py."""
# pylint: disable=missing-function-docstring,no-self-use,unnecessary-lambda
import unittest

import chex
import gojax
import haiku as hk
import jax

import models


class ModelTestCase(unittest.TestCase):
    """Tests model.py."""

    def test_get_random_model_params(self):
        model_fn = models.get_model('random')
        self.assertIsInstance(model_fn, hk.Transformed)
        params = model_fn.init(jax.random.PRNGKey(42), gojax.new_states(batch_size=2, board_size=3))
        self.assertIsInstance(params, dict)
        self.assertEqual(len(params), 0)

    def test_get_random_model_output_shape(self):
        model_fn = models.get_model('random')
        new_states = gojax.new_states(batch_size=1, board_size=3)
        params = model_fn.init(jax.random.PRNGKey(42), new_states)
        action_logits, value_logits = model_fn.apply(params, jax.random.PRNGKey(42), new_states)
        chex.assert_shape((action_logits, value_logits), ((1, 10), (1,)))

    def test_get_linear_model_params(self):
        model_fn = models.get_model('linear')
        self.assertIsInstance(model_fn, hk.Transformed)
        params = model_fn.init(jax.random.PRNGKey(42), gojax.new_states(batch_size=2, board_size=3))
        self.assertIsInstance(params, dict)
        self.assertIn('linear_go_model', params)
        self.assertEqual(len(params['linear_go_model']), 3)

    def test_get_linear_model_output_shape(self):
        model_fn = models.get_model('linear')
        new_states = gojax.new_states(batch_size=1, board_size=3)
        params = model_fn.init(jax.random.PRNGKey(42), new_states)
        action_logits, value_logits = model_fn.apply(params, jax.random.PRNGKey(42), new_states)
        chex.assert_shape((action_logits, value_logits), ((1, 10), (1,)))

    def test_get_unknown_model(self):
        model_fn = models.get_model('foo')
        with self.assertRaises(KeyError):
            model_fn.init(jax.random.PRNGKey(42),
                          gojax.new_states(batch_size=2, board_size=3))


if __name__ == '__main__':
    unittest.main()
