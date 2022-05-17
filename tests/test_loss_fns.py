"""Tests the loss functions in train.py."""
# pylint: disable=missing-function-docstring,no-self-use,unnecessary-lambda,no-value-for-parameter
import unittest

import chex
import gojax
import jax
import numpy as np
from absl.testing import parameterized
from jax import numpy as jnp

import models
import train


class LossFunctionsTestCase(chex.TestCase):
    """Test policy loss under various inputs"""

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ('zeros', [[0, 0]], [[0, 0]], 0.693147),
        ('ones', [[1, 1]], [[1, 1]], 0.693147),
        ('zero_one_one_zero', [[0, 1]], [[1, 0]], 1.04432),
        ('zero_one', [[0, 1]], [[0, 1]], 0.582203),
        # Average of 0.693147 and 0.582203
        ('batch_size_two', [[1, 1], [0, 1]], [[1, 1], [0, 1]], 0.637675),
        ('three_logits_correct', [[0, 1, 0]], [[0, 1, 0]], 0.975328),
        ('three_logits_correct', [[0, 0, 1]], [[0, 0, 1]], 0.975328),
        ('cold_temperature', [[0, 0, 1]], [[0, 0, 1]], 0.764459, 0.5),
        ('hot_temperature', [[0, 0, 1]], [[0, 0, 1]], 1.099582, 2),
        ('scale_logits', [[0, 0, 1]], [[0, 0, 2]], 0.764459),  # Same as cold temperature
    )
    def test_policy_loss_(self, action_logits, transition_value_logits, expected_loss,
                          temp=None):
        np.testing.assert_allclose(
            self.variant(train.compute_policy_loss)(jnp.array(action_logits),
                                                    jnp.array(transition_value_logits),
                                                    temp),
            expected_loss, rtol=1e-6)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ('zero_tie', [0], [0.5], 0.693147),
        ('one_tie', [1], [0.5], 0.813262),
        ('neg_one_tie', [-1], [0.5], 0.813262),
        ('zero_black', [0], [0], 0.693147),
        ('zero_white', [0], [1], 0.693147),
        ('one_black', [1], [0], 1.313262),
        ('ones', [1], [1], 0.313262),
        ('batch_size_two', [0, 1], [1, 0], 1.003204),  # Average of 0.693147 and 1.313262
        ('neg_one_black', [-1], [0], 0.313262),
        ('neg_two_black', [-2], [0], 0.126928),
    )
    def test_value_loss(self, value_logits, labels, expected_loss):
        np.testing.assert_allclose(
            self.variant(train.sigmoid_cross_entropy)(jnp.array(value_logits),
                                                      jnp.array(labels)),
            expected_loss, rtol=1e-6)


class KStepLossFnTestCase(chex.TestCase):
    """Tests compute_k_step_losses under the linear model."""

    @chex.variants(with_jit=True)
    @parameterized.named_parameters(
        # Model outputs zero logits.
        ('zeros_params_ones_state_zeros_label', 0, 1, 0, 0.6931471806),
        ('zeros_params_ones_state_ones_label', 0, 1, 1, 0.6931471806),
        ('zeros_params_ones_state_neg_ones_label', 0, 1, -1, 0.6931471806),

        ('ones_params_ones_state_zeros_label', 1, 1, 0, 27.5),  # High loss
        ('ones_params_ones_state_ones_label', 1, 1, 1, 1.2995815e-24),  # Low loss
        ('ones_params_ones_state_neg_ones_label', 1, 1, -1, 55),  # Very high loss
    )
    def test_(self, param_fill_value, state_fill_value, label_fill_value, expected_loss):
        board_size = 3
        linear_model = models.make_model(board_size, 'identity', 'linear', 'linear', 'real')
        states = jnp.full_like(gojax.new_states(batch_size=1, board_size=board_size),
                               state_fill_value)
        params = linear_model.init(jax.random.PRNGKey(42), states)
        params = jax.tree_map(lambda p: jnp.full_like(p, param_fill_value), params)
        loss_fn = self.variant(train.compute_k_step_losses, static_argnums=0)
        self.assertAlmostEqual(
            loss_fn(linear_model, params, states, actions=jnp.array((-1), dtype=int),
                    game_winners=jnp.full(len(states), label_fill_value)),
            expected_loss)


if __name__ == '__main__':
    unittest.main()
