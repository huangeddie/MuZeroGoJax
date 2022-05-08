"""Tests train.py."""
# pylint: disable=missing-function-docstring,no-self-use,unnecessary-lambda
import copy
import unittest

import chex
import gojax
import jax
import jax.numpy as jnp
from absl.testing import parameterized

import models
import train


class ValueLossFnLinearTestCase(chex.TestCase):
    """Tests value_loss_fn under the linear model."""

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        # Model outputs zero logits.
        ('_zeros_params_ones_state_zeros_label', 0, 1, 0, 0.6931471806),
        ('_zeros_params_ones_state_ones_label', 0, 1, 1, 0.6931471806),
        ('_zeros_params_ones_state_neg_ones_label', 0, 1, -1, 0.6931471806),

        ('_ones_params_ones_state_zeros_label', 1, 1, 0, 27.5),  # High loss
        ('_ones_params_ones_state_ones_label', 1, 1, 1, 1.2995815e-24),  # Low loss
        ('_ones_params_ones_state_neg_ones_label', 1, 1, -1, 55),  # Very high loss
    )
    def test(self, param_fill_value, state_fill_value, label_fill_value, expected_loss):
        linear_model = models.get_model('linear')
        states = jnp.full_like(gojax.new_states(batch_size=1, board_size=3), state_fill_value)
        params = linear_model.init(jax.random.PRNGKey(42), states)
        linear_params = params['linear_go_model']
        linear_params['value_w'] = jnp.full_like(linear_params['value_w'], param_fill_value)
        linear_params['value_b'] = jnp.full_like(linear_params['value_b'], param_fill_value)
        loss_fn = self.variant(jax.tree_util.Partial(train.value_loss_fn, linear_model))
        self.assertAlmostEqual(loss_fn(params, states, jnp.full(len(states), label_fill_value)),
                               expected_loss)


class LinearValueStepParameterizedTestCase(chex.TestCase):
    """Tests the value step under the linear model."""

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        {'testcase_name': '_zeros_params_with_single_empty_state', 'params_fill_value': 0,
         'state': gojax.new_states(batch_size=1, board_size=3),
         'expected_value_w': jnp.zeros((gojax.NUM_CHANNELS, 3, 3)), 'expected_value_b': 0},
        {'testcase_name': '_ones_params_with_single_empty_state', 'params_fill_value': 1,
         'state': gojax.new_states(batch_size=1, board_size=3),
         'expected_value_w': jnp.ones((gojax.NUM_CHANNELS, 3, 3)), 'expected_value_b': 0.768941},
        {'testcase_name': '_ones_params_with_single_black_piece', 'params_fill_value': 1,
         'state': gojax.decode_state("""
                                    _ _ _
                                    _ B _
                                    _ _ _
                                    """),
         'expected_value_w': jnp.ones((gojax.NUM_CHANNELS, 3, 3)).at[
             gojax.BLACK_CHANNEL_INDEX, 1, 1].set(1.047426).at[
             gojax.INVALID_CHANNEL_INDEX, 1, 1].set(1.047426), 'expected_value_b': 1.047426},
        {'testcase_name': '_ones_params_with_single_white_piece', 'params_fill_value': 1,
         'state': gojax.decode_state("""
                                    _ _ _
                                    _ W _
                                    _ _ _
                                    """),
         'expected_value_w': jnp.ones((gojax.NUM_CHANNELS, 3, 3)).at[
             gojax.WHITE_CHANNEL_INDEX, 1, 1].set(0.047425866).at[
             gojax.INVALID_CHANNEL_INDEX, 1, 1].set(0.047425866), 'expected_value_b': 0.047425866},
    )
    def test(self, params_fill_value, state, expected_value_w, expected_value_b):
        linear_model = models.get_model('linear')
        params = linear_model.init(jax.random.PRNGKey(42), state)
        linear_params = params['linear_go_model']
        linear_params['value_w'] = jnp.full_like(linear_params['value_w'], params_fill_value)
        linear_params['value_b'] = jnp.full_like(linear_params['value_b'], params_fill_value)

        value_step_fn = self.variant(jax.tree_util.Partial(train.value_step, linear_model))
        new_params = value_step_fn(params, jnp.expand_dims(state, axis=1), learning_rate=1)

        expected_params = copy.copy(params)
        expected_params['linear_go_model']['value_w'] = expected_value_w
        expected_params['linear_go_model']['value_b'] = expected_value_b
        chex.assert_trees_all_close(new_params, expected_params)


class TrainTestCase(unittest.TestCase):
    """Tests train.py."""

    def test_value_loss_gradient_ones_linear_with_ones_input_and_tie_labels(self):
        linear_model = models.get_model('linear')
        states = jnp.ones_like(gojax.new_states(batch_size=1, board_size=3))
        params = linear_model.init(jax.random.PRNGKey(42), states)
        linear_params = params['linear_go_model']
        linear_params['value_w'] = jnp.ones_like(linear_params['value_w'])
        linear_params['value_b'] = jnp.ones_like(linear_params['value_b'])

        grads = jax.grad(jax.tree_util.Partial(train.value_loss_fn, linear_model))(params, states,
                                                                                   jnp.zeros(
                                                                                       len(states)))

        # Positive gradient for only value parameters.
        expected_grad = copy.copy(params)
        expected_grad['linear_go_model']['value_w'] = jnp.full_like(
            jnp.zeros_like(params['linear_go_model']['value_w']),
            fill_value=0.5)
        expected_grad['linear_go_model']['value_b'] = 0.5
        # No gradient for other parameters.
        expected_grad['linear_go_model']['action_w'] = jnp.zeros_like(
            params['linear_go_model']['action_w'])
        expected_grad['linear_go_model']['transition_w'] = jnp.zeros_like(
            params['linear_go_model']['transition_w'])
        expected_grad['linear_go_model']['transition_b'] = jnp.zeros_like(
            params['linear_go_model']['transition_b'])

        chex.assert_trees_all_close(grads, expected_grad)

    if __name__ == '__main__':
        unittest.main()
