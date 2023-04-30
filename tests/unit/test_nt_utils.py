"""Test nt_utils.py"""
# pylint: disable=missing-function-docstring,no-value-for-parameter,too-many-public-methods,duplicate-code
import unittest

import chex
import jax.random
import numpy as np
from absl.testing import parameterized
from jax import numpy as jnp

from muzero_gojax import nt_utils


class NtUtilsTestCase(chex.TestCase):
    """Test nt_utils.py"""

    def test_flatten_first_two_dims_on_ranks_2_3_and_4(self):
        chex.assert_shape(nt_utils.flatten_first_two_dims(jnp.zeros((3, 4))),
                          (12, ))
        chex.assert_shape(
            nt_utils.flatten_first_two_dims(jnp.zeros((3, 4, 5))), (12, 5))
        chex.assert_shape(
            nt_utils.flatten_first_two_dims(jnp.zeros((3, 4, 5, 6))),
            (12, 5, 6))

    def test_unflatten_first_dim_on_ranks_2_3_and_4(self):
        chex.assert_shape(nt_utils.unflatten_first_dim(jnp.zeros((12)), 3, 4),
                          (
                              3,
                              4,
                          ))
        chex.assert_shape(
            nt_utils.unflatten_first_dim(jnp.zeros((12, 5)), 3, 4), (3, 4, 5))
        chex.assert_shape(
            nt_utils.unflatten_first_dim(jnp.zeros((12, 5, 6)), 3, 4),
            (3, 4, 5, 6))

    @parameterized.named_parameters(
        dict(testcase_name='zero',
             batch_size=1,
             total_steps=1,
             k=0,
             expected_output=[[False]]),
        dict(testcase_name='one',
             batch_size=1,
             total_steps=1,
             k=1,
             expected_output=[[True]]),
        dict(testcase_name='zeros',
             batch_size=1,
             total_steps=2,
             k=0,
             expected_output=[[False, False]]),
        dict(testcase_name='half',
             batch_size=1,
             total_steps=2,
             k=1,
             expected_output=[[True, False]]),
        dict(testcase_name='full',
             batch_size=1,
             total_steps=2,
             k=2,
             expected_output=[[True, True]]),
        dict(testcase_name='batch_size_2_zero',
             batch_size=2,
             total_steps=1,
             k=0,
             expected_output=[[False], [False]]),
        dict(testcase_name='batch_size_2_one',
             batch_size=2,
             total_steps=1,
             k=1,
             expected_output=[[True], [True]]),
        dict(testcase_name='batch_size_2_zeros',
             batch_size=2,
             total_steps=2,
             k=0,
             expected_output=[[False, False], [False, False]]),
        dict(testcase_name='batch_size_2_half',
             batch_size=2,
             total_steps=2,
             k=1,
             expected_output=[[True, False], [True, False]]),
        dict(testcase_name='batch_size_2_full',
             batch_size=2,
             total_steps=2,
             k=2,
             expected_output=[[True, True], [True, True]]),
    )
    def test_make_prefix_nt_mask(self, batch_size, total_steps, k,
                                 expected_output):
        """Tests the make_prefix_nt_mask based on inputs and expected output."""
        np.testing.assert_array_equal(
            nt_utils.make_prefix_nt_mask(batch_size, total_steps, k),
            expected_output)

    @parameterized.named_parameters(
        dict(testcase_name='zero',
             batch_size=1,
             total_steps=1,
             k=0,
             expected_output=[[False]]),
        dict(testcase_name='one',
             batch_size=1,
             total_steps=1,
             k=1,
             expected_output=[[True]]),
        dict(testcase_name='zeros',
             batch_size=1,
             total_steps=2,
             k=0,
             expected_output=[[False, False]]),
        dict(testcase_name='half',
             batch_size=1,
             total_steps=2,
             k=1,
             expected_output=[[False, True]]),
        dict(testcase_name='full',
             batch_size=1,
             total_steps=2,
             k=2,
             expected_output=[[True, True]]),
        dict(testcase_name='batch_size_2_zero',
             batch_size=2,
             total_steps=1,
             k=0,
             expected_output=[[False], [False]]),
        dict(testcase_name='batch_size_2_one',
             batch_size=2,
             total_steps=1,
             k=1,
             expected_output=[[True], [True]]),
        dict(testcase_name='batch_size_2_zeros',
             batch_size=2,
             total_steps=2,
             k=0,
             expected_output=[[False, False], [False, False]]),
        dict(testcase_name='batch_size_2_half',
             batch_size=2,
             total_steps=2,
             k=1,
             expected_output=[[False, True], [False, True]]),
        dict(testcase_name='batch_size_2_full',
             batch_size=2,
             total_steps=2,
             k=2,
             expected_output=[[True, True], [True, True]]),
    )
    def make_suffix_nt_mask(self, batch_size, total_steps, k, expected_output):
        """Tests the make_prefix_nt_mask based on inputs and expected output."""
        np.testing.assert_array_equal(
            nt_utils.make_suffix_nt_mask(batch_size, total_steps, k),
            expected_output)

    @parameterized.named_parameters(
        dict(testcase_name='all_zeros',
             x_logits=[[0, 0]],
             y_logits=[[0, 0]],
             expected_loss=0),
        dict(testcase_name='all_ones',
             x_logits=[[1, 1]],
             y_logits=[[1, 1]],
             expected_loss=0),
        dict(testcase_name='different_logits',
             x_logits=[[0, 1]],
             y_logits=[[1, 0]],
             expected_loss=0.462117),
        dict(testcase_name='similar_logits',
             x_logits=[[0, 1]],
             y_logits=[[0, 1]],
             expected_loss=0),
        dict(testcase_name='batch_size_two_similar_logits',
             x_logits=[[1, 1], [0, 1]],
             y_logits=[[1, 1], [0, 1]],
             expected_loss=0),
        dict(testcase_name='similar_3d_logits',
             x_logits=[[0, 1, 0]],
             y_logits=[[0, 1, 0]],
             expected_loss=0),
        dict(testcase_name='similar_3d_logits_v2',
             x_logits=[[0, 0, 1]],
             y_logits=[[0, 0, 1]],
             expected_loss=0),
        dict(testcase_name='amplified_target_logits',
             x_logits=[[0, 0, 1]],
             y_logits=[[0, 0, 2]],
             expected_loss=0.098886),
    )
    def test_nt_categorical_kl_divergence(self, x_logits, y_logits,
                                          expected_loss):
        """Tests the nt_categorical_kl_divergence."""
        np.testing.assert_allclose(nt_utils.nt_categorical_kl_divergence(
            jnp.array(x_logits), jnp.array(y_logits)),
                                   expected_loss,
                                   rtol=1e-5)

    @parameterized.named_parameters(
        dict(testcase_name='low_entropy',
             logits=[[10, 0]],
             expected_entropy=0.000499),
        dict(testcase_name='high_entropy',
             logits=[[0, 0]],
             expected_entropy=0.693147),
        dict(testcase_name='batch_size_two_mixed_entropy',
             logits=[[100, 0], [0, 0]],
             expected_entropy=0.346574))
    def test_nt_entropy(self, logits, expected_entropy):
        np.testing.assert_allclose(nt_utils.nt_entropy(jnp.array(logits)),
                                   expected_entropy,
                                   rtol=1e-3)

    def test_nt_categorical_kl_divergence_gradient_of_identical_logits_are_near_zero(
            self):
        random_nt_array = jax.random.uniform(jax.random.PRNGKey(42),
                                             (2, 3, 4)) + 1
        grad = jax.grad(lambda y: nt_utils.nt_categorical_kl_divergence(
            y, jax.lax.stop_gradient(y)))(random_nt_array)
        np.testing.assert_allclose(grad, jnp.zeros_like(grad), atol=1e-6)

    def test_nt_categorical_kl_divergence_gradient_of_identical_bfloat16_logits_are_less_near_zero(
            self):
        random_nt_array = jax.random.uniform(jax.random.PRNGKey(42), (2, 3, 4),
                                             dtype='bfloat16')
        grad = jax.grad(lambda y: nt_utils.nt_categorical_kl_divergence(
            y, jax.lax.stop_gradient(y)))(random_nt_array).astype(float)
        np.testing.assert_allclose(grad, jnp.zeros_like(grad), atol=1e-3)

    @parameterized.named_parameters(
        dict(testcase_name='mid_logit_mid_label',
             logits=[0],
             labels=[0.5],
             expected_loss=0.693147),
        dict(testcase_name='high_logit_mid_label',
             logits=[1],
             labels=[0.5],
             expected_loss=0.813262),
        dict(testcase_name='low_logit_mid_label',
             logits=[-1],
             labels=[0.5],
             expected_loss=0.813262),
        dict(testcase_name='mid_logit_low_label',
             logits=[0],
             labels=[0],
             expected_loss=0.693147),
        dict(testcase_name='mid_logit_high_label',
             logits=[0],
             labels=[1],
             expected_loss=0.693147),
        dict(testcase_name='high_logit_low_label',
             logits=[1],
             labels=[0],
             expected_loss=1.313262),
        dict(testcase_name='high_logit_high_label',
             logits=[1],
             labels=[1],
             expected_loss=0.313262),
        dict(testcase_name='low_logit_low_label',
             logits=[-1],
             labels=[0],
             expected_loss=0.313262),
        dict(testcase_name='very_low_logit_low_label',
             logits=[-2],
             labels=[0],
             expected_loss=0.126928),  # Average of 0.693147 and 1.313262
        dict(testcase_name='batch_size_two_averages_loss',
             logits=[0, 1],
             labels=[1, 0],
             expected_loss=1.003204),
        dict(testcase_name='nt_logits_averages_everything',
             logits=jnp.ones((2, 2, 4, 4)),
             labels=jnp.zeros((2, 2, 4, 4)),
             expected_loss=1.313262))
    def test_sigmoid_cross_entropy(self, logits, labels, expected_loss):
        """Tests the nt_sigmoid_cross_entropy."""
        np.testing.assert_allclose(nt_utils.nt_sigmoid_cross_entropy(
            jnp.array(logits), jnp.array(labels)),
                                   expected_loss,
                                   rtol=1e-6)

    def test_kl_div_trans_loss_with_full_mask_returns_positive_value(self):
        transitions = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 2))
        expected_transitions = jax.random.normal(jax.random.PRNGKey(69),
                                                 (2, 2, 2))

        kl_div = nt_utils.nt_kl_div_loss(transitions, expected_transitions,
                                         nt_utils.make_prefix_nt_mask(
                                             2, 2, 2)).item()

        self.assertGreater(kl_div, 0)

    def test_kl_div_trans_loss_with_half_mask_returns_positive_value(self):
        transitions = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 2))
        expected_transitions = jax.random.normal(jax.random.PRNGKey(69),
                                                 (2, 2, 2))

        kl_div = nt_utils.nt_kl_div_loss(transitions, expected_transitions,
                                         nt_utils.make_prefix_nt_mask(2, 2, 1))

        self.assertGreater(kl_div, 0)

    def test_mse_trans_loss_with_full_mask_returns_positive_value(self):
        transitions = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 2))
        expected_transitions = jax.random.normal(jax.random.PRNGKey(69),
                                                 (2, 2, 2))

        mse = nt_utils.nt_mse_loss(transitions, expected_transitions,
                                   nt_utils.make_prefix_nt_mask(2, 2,
                                                                2)).item()

        self.assertGreater(mse, 0)

    def test_mse_trans_loss_with_half_mask_returns_positive_value(self):
        transitions = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 2))
        expected_transitions = jax.random.normal(jax.random.PRNGKey(69),
                                                 (2, 2, 2))

        mse = nt_utils.nt_mse_loss(transitions, expected_transitions,
                                   nt_utils.make_prefix_nt_mask(2, 2,
                                                                1)).item()

        self.assertGreater(mse, 0)

    def test_bce_trans_loss_with_full_mask_returns_positive_value(self):
        transitions = jax.random.uniform(jax.random.PRNGKey(42), (2, 2, 2))
        expected_transitions = jax.random.bernoulli(
            jax.random.PRNGKey(69), shape=(2, 2, 2)).astype('bfloat16')

        bce = nt_utils.nt_bce_loss(transitions, expected_transitions,
                                   nt_utils.make_prefix_nt_mask(2, 2,
                                                                2)).item()

        self.assertGreater(bce, 0)

    def test_bce_trans_loss_with_half_mask_returns_positive_value(self):
        transitions = jax.random.uniform(jax.random.PRNGKey(42), (2, 2, 2))
        expected_transitions = jax.random.bernoulli(
            jax.random.PRNGKey(69), shape=(2, 2, 2)).astype('bfloat16')

        bce = nt_utils.nt_bce_loss(transitions, expected_transitions,
                                   nt_utils.make_prefix_nt_mask(2, 2,
                                                                1)).item()

        self.assertGreater(bce, 0)

    def test_bfloat16_bce_trans_loss_with_extreme_values_returns_finite_value(
            self):
        transitions = jnp.array([[[1]]], dtype='bfloat16')
        expected_transitions = jax.random.bernoulli(
            jax.random.PRNGKey(69), shape=(1, 1, 1)).astype('bfloat16')

        self.assertTrue(
            np.isfinite(
                nt_utils.nt_bce_loss(transitions, expected_transitions,
                                     nt_utils.make_prefix_nt_mask(1, 1, 1))))

    def test_sign_trans_acc_with_full_mask_is_within_range(self):
        transitions = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 2))
        expected_transitions = jax.random.normal(jax.random.PRNGKey(69),
                                                 shape=(2, 2,
                                                        2)).astype('bfloat16')

        sign_acc = nt_utils.nt_sign_acc(transitions, expected_transitions,
                                        nt_utils.make_prefix_nt_mask(
                                            2, 2, 2)).item()
        self.assertBetween(sign_acc, 0, 1)

    def test_sign_trans_acc_with_half_mask_is_within_range(self):
        transitions = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 2))
        expected_transitions = jax.random.normal(jax.random.PRNGKey(69),
                                                 shape=(2, 2,
                                                        2)).astype('bfloat16')

        sign_acc = nt_utils.nt_sign_acc(transitions, expected_transitions,
                                        nt_utils.make_prefix_nt_mask(
                                            2, 2, 1)).item()
        self.assertBetween(sign_acc, 0, 1)


if __name__ == '__main__':
    unittest.main()
