"""Test nt_utils.py"""
# pylint: disable=missing-function-docstring,no-self-use,no-value-for-parameter,too-many-public-methods,duplicate-code
import unittest

import chex
import jax.random
import numpy as np
from absl.testing import parameterized
from jax import numpy as jnp

from muzero_gojax import nt_utils


class NtUtilsTestCase(chex.TestCase):
    """Test nt_utils.py"""

    @parameterized.named_parameters(('zero', 1, 1, 0, [[False]]), ('one', 1, 1, 1, [[True]]),
                                    ('zeros', 1, 2, 0, [[False, False]]),
                                    ('half', 1, 2, 1, [[True, False]]),
                                    ('full', 1, 2, 2, [[True, True]]),
                                    ('b2_zero', 2, 1, 0, [[False], [False]]),
                                    ('b2_one', 2, 1, 1, [[True], [True]]),
                                    ('b2_zeros', 2, 2, 0, [[False, False], [False, False]]),
                                    ('b2_half', 2, 2, 1, [[True, False], [True, False]]),
                                    ('b2_full', 2, 2, 2, [[True, True], [True, True]]), )
    def test_prefix_k_steps_mask(self, batch_size, total_steps, k, expected_output):
        """Tests the make_prefix_nt_mask based on inputs and expected output."""
        np.testing.assert_array_equal(nt_utils.make_prefix_nt_mask(batch_size, total_steps, k),
                                      expected_output)

    @parameterized.named_parameters(('zero', 1, 1, 0, [[False]]), ('one', 1, 1, 1, [[True]]),
                                    ('zeros', 1, 2, 0, [[False, False]]),
                                    ('half', 1, 2, 1, [[False, True]]),
                                    ('full', 1, 2, 2, [[True, True]]),
                                    ('b2_zero', 2, 1, 0, [[False], [False]]),
                                    ('b2_one', 2, 1, 1, [[True], [True]]),
                                    ('b2_zeros', 2, 2, 0, [[False, False], [False, False]]),
                                    ('b2_half', 2, 2, 1, [[False, True], [False, True]]),
                                    ('b2_full', 2, 2, 2, [[True, True], [True, True]]), )
    def test_suffix_k_steps_mask(self, batch_size, total_steps, k, expected_output):
        """Tests the make_prefix_nt_mask based on inputs and expected output."""
        np.testing.assert_array_equal(nt_utils.make_suffix_nt_mask(batch_size, total_steps, k),
                                      expected_output)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(('zeros', [[0, 0]], [[0, 0]], 0.693147),
                                    ('ones', [[1, 1]], [[1, 1]], 0.693147),
                                    ('zero_one_one_zero', [[0, 1]], [[1, 0]], 1.04432),
                                    ('zero_one', [[0, 1]], [[0, 1]], 0.582203),
                                    # Average of 0.693147 and 0.582203
                                    (
                                    'batch_size_two', [[1, 1], [0, 1]], [[1, 1], [0, 1]], 0.637675),
                                    ('three_logits_correct', [[0, 1, 0]], [[0, 1, 0]], 0.975328),
                                    ('three_logits_correct', [[0, 0, 1]], [[0, 0, 1]], 0.975328),
                                    ('cold_temperature', [[0, 0, 1]], [[0, 0, 1]], 0.764459, 0.5),
                                    ('hot_temperature', [[0, 0, 1]], [[0, 0, 1]], 1.099582, 2),
                                    ('scale_logits', [[0, 0, 1]], [[0, 0, 2]], 0.764459),
                                    # Same as cold temperature
                                    )
    def test_nt_categorical_cross_entropy(self, action_logits, transition_value_logits,
                                          expected_loss, temp=None):
        """Tests the nt_categorical_cross_entropy."""
        np.testing.assert_allclose(
            self.variant(nt_utils.nt_categorical_cross_entropy)(jnp.array(action_logits),
                                                                jnp.array(transition_value_logits),
                                                                temp), expected_loss, rtol=1e-6)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(('zero_tie', [0], [0.5], 0.693147),
                                    ('one_tie', [1], [0.5], 0.813262),
                                    ('neg_one_tie', [-1], [0.5], 0.813262),
                                    ('zero_black', [0], [0], 0.693147),
                                    ('zero_white', [0], [1], 0.693147),
                                    ('one_black', [1], [0], 1.313262), ('ones', [1], [1], 0.313262),
                                    ('batch_size_two', [0, 1], [1, 0], 1.003204),
                                    # Average of 0.693147 and 1.313262
                                    ('neg_one_black', [-1], [0], 0.313262),
                                    ('neg_two_black', [-2], [0], 0.126928), )
    def test_sigmoid_cross_entropy(self, value_logits, labels, expected_loss):
        """Tests the nt_sigmoid_cross_entropy."""
        np.testing.assert_allclose(
            self.variant(nt_utils.nt_sigmoid_cross_entropy)(jnp.array(value_logits),
                                                            jnp.array(labels)), expected_loss,
            rtol=1e-6)

    def test_kl_div_trans_loss_with_full_mask(self):
        transitions = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 2))
        expected_transitions = jax.random.normal(jax.random.PRNGKey(69), (2, 2, 2))
        np.testing.assert_allclose(nt_utils.nt_kl_div_loss(transitions, expected_transitions,
                                                           nt_utils.make_prefix_nt_mask(2, 2, 2)),
                                   0.464844, atol=1e-5)

    def test_kl_div_trans_loss_with_half_mask(self):
        transitions = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 2))
        expected_transitions = jax.random.normal(jax.random.PRNGKey(69), (2, 2, 2))
        np.testing.assert_allclose(nt_utils.nt_kl_div_loss(transitions, expected_transitions,
                                                           nt_utils.make_prefix_nt_mask(2, 2, 1)),
                                   0.777344, atol=1e-5)

    def test_mse_trans_loss_with_full_mask(self):
        transitions = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 2))
        expected_transitions = jax.random.normal(jax.random.PRNGKey(69), (2, 2, 2))
        np.testing.assert_allclose(nt_utils.nt_mse_loss(transitions, expected_transitions,
                                                        nt_utils.make_prefix_nt_mask(2, 2, 2)),
                                   3.718629, atol=1e-5)

    def test_mse_trans_loss_with_half_mask(self):
        transitions = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 2))
        expected_transitions = jax.random.normal(jax.random.PRNGKey(69), (2, 2, 2))
        np.testing.assert_allclose(nt_utils.nt_mse_loss(transitions, expected_transitions,
                                                        nt_utils.make_prefix_nt_mask(2, 2, 1)),
                                   7.082739, atol=1e-5)

    def test_bce_trans_loss_with_full_mask(self):
        transitions = jax.random.uniform(jax.random.PRNGKey(42), (2, 2, 2))
        expected_transitions = jax.random.bernoulli(jax.random.PRNGKey(69), shape=(2, 2, 2)).astype(
            'bfloat16')
        np.testing.assert_allclose(nt_utils.nt_bce_loss(transitions, expected_transitions,
                                                        nt_utils.make_prefix_nt_mask(2, 2, 2)),
                                   1.296682, atol=1e-5)

    def test_bce_trans_loss_with_half_mask(self):
        transitions = jax.random.uniform(jax.random.PRNGKey(42), (2, 2, 2))
        expected_transitions = jax.random.bernoulli(jax.random.PRNGKey(69), shape=(2, 2, 2)).astype(
            'bfloat16')
        np.testing.assert_allclose(nt_utils.nt_bce_loss(transitions, expected_transitions,
                                                        nt_utils.make_prefix_nt_mask(2, 2, 1)),
                                   1.336445, atol=1e-5)

    def test_bce_trans_loss_with_extreme_values(self):
        transitions = jnp.array([[[1]]], dtype='bfloat16')
        expected_transitions = jax.random.bernoulli(jax.random.PRNGKey(69), shape=(1, 1, 1)).astype(
            'bfloat16')
        self.assertTrue(np.isfinite(nt_utils.nt_bce_loss(transitions, expected_transitions,
                                                         nt_utils.make_prefix_nt_mask(1, 1, 1))))

    def test_bce_trans_acc_with_full_mask(self):
        transitions = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 2))
        expected_transitions = jax.random.bernoulli(jax.random.PRNGKey(69), shape=(2, 2, 2)).astype(
            'bfloat16')
        np.testing.assert_allclose(nt_utils.nt_bce_logits_acc(transitions, expected_transitions,
                                                              nt_utils.make_prefix_nt_mask(2, 2,
                                                                                           2)),
                                   0.375, atol=1e-5)

    def test_bce_trans_acc_with_half_mask(self):
        transitions = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 2))
        expected_transitions = jax.random.bernoulli(jax.random.PRNGKey(69), shape=(2, 2, 2)).astype(
            'bfloat16')
        np.testing.assert_allclose(nt_utils.nt_bce_logits_acc(transitions, expected_transitions,
                                                              nt_utils.make_prefix_nt_mask(2, 2,
                                                                                           1)),
                                   0.75, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
