"""Tests the loss functions in train.py."""
# pylint: disable=missing-function-docstring,no-self-use,unnecessary-lambda,no-value-for-parameter
import unittest
from unittest import mock

import chex
import gojax
import numpy as np
from absl.testing import parameterized
from jax import numpy as jnp

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

    @parameterized.named_parameters(
        {'testcase_name': 'zero_output_tie_game',
         'state_embed_output': [[[0.]]], 'value_output': [[0.]],
         'second_value_output': [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
         'policy_output': jnp.zeros((1, 10)), 'transition_output': jnp.zeros((1, 10, 1)),
         'game_winners': [[0]], 'expected_value_loss': 0.6931471806},
        {'testcase_name': 'zero_output_black_wins',
         'state_embed_output': [[[0.]]], 'value_output': [[0.]],
         'second_value_output': [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
         'policy_output': jnp.zeros((1, 10)), 'transition_output': jnp.zeros((1, 10, 1)),
         'game_winners': [[1]], 'expected_value_loss': 0.6931471806},
        {'testcase_name': 'zero_output_white_wins',
         'state_embed_output': [[[0.]]], 'value_output': [[0.]],
         'second_value_output': [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
         'policy_output': jnp.zeros((1, 10)), 'transition_output': jnp.zeros((1, 10, 1)),
         'game_winners': [[1]], 'expected_value_loss': 0.6931471806},
        {'testcase_name': 'one_output_tie_game',
         'state_embed_output': [[[1.]]], 'value_output': [[1.]],
         'second_value_output': [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]],
         'policy_output': jnp.ones((1, 10)), 'transition_output': jnp.ones((1, 10, 1)),
         'game_winners': [[0]], 'expected_value_loss': 0.81326175},
        {'testcase_name': 'one_output_black_wins',
         'state_embed_output': [[[1.]]], 'value_output': [[1.]],
         'second_value_output': [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]],
         'policy_output': jnp.ones((1, 10)), 'transition_output': jnp.ones((1, 10, 1)),
         'game_winners': [[1]], 'expected_value_loss': 0.3132617},
        {'testcase_name': 'one_output_white_wins',
         'state_embed_output': [[[1.]]], 'value_output': [[1.]],
         'second_value_output': [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]],
         'policy_output': jnp.ones((1, 10)), 'transition_output': jnp.ones((1, 10, 1)),
         'game_winners': [[-1]], 'expected_value_loss': 1.3132617},
    )
    def test_single_state_(self, state_embed_output, value_output, second_value_output,
                           policy_output, transition_output,
                           game_winners, expected_value_loss):
        # Build the mock model.
        mock_model = mock.Mock()
        state_embed_mock_model = mock.Mock(return_value=jnp.array(state_embed_output))
        value_mock_model = mock.Mock(
            side_effect=[jnp.array(value_output), jnp.array(second_value_output)] * 2)
        policy_mock_model = mock.Mock(return_value=jnp.array(policy_output))
        transition_mock_model = mock.Mock(return_value=jnp.array(transition_output))
        mock_model.apply = (
            state_embed_mock_model,
            value_mock_model,
            policy_mock_model,
            transition_mock_model,
        )
        params = {}

        # Execute the loss function.
        loss_dict = train.compute_k_step_losses(mock_model, params,
                                                gojax.new_states(board_size=3, batch_size=1),
                                                jnp.array([[-1]]), jnp.array(game_winners))
        # Test.
        self.assertLen(loss_dict, 2)
        self.assertIn('cum_val_loss', loss_dict)
        self.assertIn('cum_policy_loss', loss_dict)
        if expected_value_loss is not None:
            self.assertAlmostEqual(loss_dict['cum_val_loss'], expected_value_loss)
        self.assertTrue(np.isnan(loss_dict['cum_policy_loss']))
        self.assertEqual(state_embed_mock_model.call_count, 1)
        # Compiles the body_fun in the for loop so the mocks below are called twice the normal
        # amount.
        self.assertEqual(value_mock_model.call_count, 4)
        self.assertEqual(policy_mock_model.call_count, 2)
        self.assertEqual(transition_mock_model.call_count, 2)


if __name__ == '__main__':
    unittest.main()
