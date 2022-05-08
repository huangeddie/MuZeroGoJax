"""Tests model.py."""
# pylint: disable=missing-function-docstring,no-self-use,unnecessary-lambda
import unittest

import chex
import gojax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

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
        action_logits, value_logits, transition_logits = model_fn.apply(params,
                                                                        jax.random.PRNGKey(42),
                                                                        new_states)
        chex.assert_shape((action_logits, value_logits), ((1, 10), (1,)))

    def test_get_linear_model_params(self):
        model_fn = models.get_model('linear')
        self.assertIsInstance(model_fn, hk.Transformed)
        params = model_fn.init(jax.random.PRNGKey(42), gojax.new_states(batch_size=2, board_size=3))
        self.assertIsInstance(params, dict)
        self.assertIn('linear_go_model', params)
        self.assertEqual(len(params['linear_go_model']), 5)
        self.assertIn('action_w', params['linear_go_model'])
        self.assertIn('value_w', params['linear_go_model'])
        self.assertIn('value_b', params['linear_go_model'])

    def test_get_linear_model_output_shape(self):
        model_fn = models.get_model('linear')
        new_states = gojax.new_states(batch_size=1, board_size=3)
        params = model_fn.init(jax.random.PRNGKey(42), new_states)
        action_logits, value_logits, transition_logits = model_fn.apply(params,
                                                                        jax.random.PRNGKey(42),
                                                                        new_states)
        chex.assert_shape((action_logits, value_logits), ((1, 10), (1,)))

    def test_get_linear_model_output_zero_params(self):
        model_fn = hk.without_apply_rng(models.get_model('linear'))
        new_states = gojax.new_states(batch_size=1, board_size=3)
        params = model_fn.init(jax.random.PRNGKey(42), new_states)
        linear_params = params['linear_go_model']
        linear_params['action_w'] = jnp.zeros_like(linear_params['action_w'])
        linear_params['value_w'] = jnp.zeros_like(linear_params['value_w'])
        linear_params['value_b'] = jnp.zeros_like(linear_params['value_b'])

        action_logits, value_logits, transition_logits = model_fn.apply(params,
                                                                        jnp.ones_like(new_states))
        np.testing.assert_array_equal(action_logits, jnp.zeros_like(action_logits))
        np.testing.assert_array_equal(value_logits, jnp.zeros_like(value_logits))

    def test_get_linear_model_output_ones_params(self):
        model_fn = hk.without_apply_rng(models.get_model('linear'))
        new_states = gojax.new_states(batch_size=1, board_size=3)
        params = model_fn.init(jax.random.PRNGKey(42), new_states)
        linear_params = params['linear_go_model']
        linear_params['action_w'] = jnp.ones_like(linear_params['action_w'])
        linear_params['value_w'] = jnp.ones_like(linear_params['value_w'])
        linear_params['value_b'] = jnp.ones_like(linear_params['value_b'])

        action_logits, value_logits, transition_logits = model_fn.apply(params,
                                                                        jnp.ones_like(new_states))
        np.testing.assert_array_equal(action_logits, jnp.full_like(action_logits, 6 * 3 * 3))
        np.testing.assert_array_equal(value_logits, jnp.full_like(value_logits, 6 * 3 * 3 + 1))

    def test_get_unknown_model(self):
        model_fn = models.get_model('foo')
        with self.assertRaises(KeyError):
            model_fn.init(jax.random.PRNGKey(42),
                          gojax.new_states(batch_size=2, board_size=3))


if __name__ == '__main__':
    unittest.main()
