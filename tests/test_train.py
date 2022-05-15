"""Tests train.py."""
# pylint: disable=missing-function-docstring,no-self-use,unnecessary-lambda,no-value-for-parameter
import copy
import unittest

import chex
import gojax
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

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
        ('zeros', [0], [0], 0.693147),
        ('zero_one', [0], [1], 0.693147),
        ('one_zero', [1], [0], 1.313262),
        ('ones', [1], [1], 0.313262),
        ('batch_size_two', [0, 1], [1, 0], 1.003204),  # Average of 0.693147 and 1.313262
        ('neg_one_zero', [-1], [0], 0.313262),
        ('neg_two_zero', [-2], [0], 0.186334),
    )
    def test_value_loss(self, value_logits, game_winners, expected_loss):
        np.testing.assert_allclose(
            self.variant(train.sigmoid_cross_entropy)(jnp.array(value_logits),
                                                      jnp.array(game_winners)),
            expected_loss, rtol=1e-6)


class LinearValueLossFnTestCase(chex.TestCase):
    """Tests compute_k_step_losses under the linear model."""

    @chex.variants(with_jit=True, without_jit=True)
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


class LinearValueStepTestCase(chex.TestCase):
    """Tests the value step under the linear model."""

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        {'testcase_name': 'zeros_params_with_single_empty_state', 'params_fill_value': 0,
         'state': gojax.new_states(batch_size=1, board_size=3),
         'expected_value_w': jnp.zeros((gojax.NUM_CHANNELS, 3, 3)), 'expected_value_b': 0},
        {'testcase_name': 'ones_params_with_single_empty_state', 'params_fill_value': 1,
         'state': gojax.new_states(batch_size=1, board_size=3),
         'expected_value_w': jnp.ones((gojax.NUM_CHANNELS, 3, 3)), 'expected_value_b': 0.768941},
        {'testcase_name': 'ones_params_with_single_black_piece', 'params_fill_value': 1,
         'state': gojax.decode_states("""
                                    _ _ _
                                    _ B _
                                    _ _ _
                                    """),
         'expected_value_w': jnp.ones((gojax.NUM_CHANNELS, 3, 3)).at[
             gojax.BLACK_CHANNEL_INDEX, 1, 1].set(1.047426).at[
             gojax.INVALID_CHANNEL_INDEX, 1, 1].set(1.047426), 'expected_value_b': 1.047426},
        {'testcase_name': 'ones_params_with_single_white_piece', 'params_fill_value': 1,
         'state': gojax.decode_states("""
                                    _ _ _
                                    _ W _
                                    _ _ _
                                    """),
         'expected_value_w': jnp.ones((gojax.NUM_CHANNELS, 3, 3)).at[
             gojax.WHITE_CHANNEL_INDEX, 1, 1].set(0.047425866).at[
             gojax.INVALID_CHANNEL_INDEX, 1, 1].set(0.047425866), 'expected_value_b': 0.047425866},
    )
    def test_(self, params_fill_value, state, expected_value_w, expected_value_b):
        board_size = 3
        linear_model = models.make_model(board_size, 'identity', 'linear', 'linear', 'real')
        params = linear_model.init(jax.random.PRNGKey(42), state)
        params = jax.tree_map(lambda p: jnp.full_like(p, params_fill_value), params)

        value_step_fn = self.variant(train.train_step, static_argnums=0)
        state_data, actions, game_winners = train.trajectories_to_dataset(
            jnp.expand_dims(state, axis=1))
        new_params, _ = value_step_fn(linear_model, params, state_data, actions,
                                      game_winners,
                                      learning_rate=1)

        expected_params = copy.copy(params)
        expected_params['linear3_d_value']['value_w'] = expected_value_w
        expected_params['linear3_d_value']['value_b'] = expected_value_b
        chex.assert_trees_all_close(new_params, expected_params)


class TrainTestCase(unittest.TestCase):
    """Tests train.py."""

    def test_value_loss_gradient_ones_linear_with_ones_input_and_tie_labels(self):
        board_size = 3
        linear_model = models.make_model(board_size, 'identity', 'linear', 'linear', 'real')
        states = jnp.ones_like(gojax.new_states(batch_size=1, board_size=board_size))
        actions = jnp.array((-1))
        params = linear_model.init(jax.random.PRNGKey(42), states)
        params = jax.tree_map(lambda p: jnp.ones_like(p), params)

        grads = jax.grad(train.compute_k_step_losses, argnums=1)(linear_model, params, states,
                                                                 actions,
                                                                 jnp.zeros(len(states)))

        # Positive gradient for only value parameters.
        expected_grad = copy.copy(params)
        expected_grad['linear3_d_value']['value_w'] = jnp.full_like(
            jnp.zeros_like(params['linear3_d_value']['value_w']),
            fill_value=0.5)
        expected_grad['linear3_d_value']['value_b'] = 0.5
        # No gradient for other parameters.
        expected_grad['linear3_d_policy']['action_w'] = jnp.zeros_like(
            params['linear3_d_policy']['action_w'])

        chex.assert_trees_all_close(grads, expected_grad)

    if __name__ == '__main__':
        unittest.main()
