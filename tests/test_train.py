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


class TrainTestCase(unittest.TestCase):
    """Tests train.py."""

    def test_value_loss_gradient_ones_linear_with_ones_input_and_tie_labels(self):
        linear_model = models.get_model('linear')
        states = jnp.ones_like(gojax.new_states(batch_size=1, board_size=3))
        params = linear_model.init(jax.random.PRNGKey(42), states)
        linear_params = params['linear_go_model']
        linear_params['action_w'] = jnp.ones_like(linear_params['action_w'])
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

    def test_value_step_zeros_linear_with_single_empty_state(self):
        linear_model = models.get_model('linear')
        states = gojax.new_states(batch_size=1, board_size=3)
        params = linear_model.init(jax.random.PRNGKey(42), states)
        linear_params = params['linear_go_model']
        linear_params['action_w'] = jnp.zeros_like(linear_params['action_w'])
        linear_params['value_w'] = jnp.zeros_like(linear_params['value_w'])
        linear_params['value_b'] = jnp.zeros_like(linear_params['value_b'])

        new_params = train.value_step(linear_model, params, jnp.expand_dims(states, axis=1),
                                      learning_rate=0.1)

        chex.assert_trees_all_equal(new_params, params)

    def test_value_step_ones_linear_with_single_empty_state(self):
        linear_model = models.get_model('linear')
        states = gojax.new_states(batch_size=1, board_size=3)
        params = linear_model.init(jax.random.PRNGKey(42), states)
        linear_params = params['linear_go_model']
        linear_params['action_w'] = jnp.ones_like(linear_params['action_w'])
        linear_params['value_w'] = jnp.ones_like(linear_params['value_w'])
        linear_params['value_b'] = jnp.ones_like(linear_params['value_b'])

        new_params = train.value_step(linear_model, params, jnp.expand_dims(states, axis=1),
                                      learning_rate=1)

        # Decrease value parameters towards 0.
        expected_params = copy.copy(params)
        expected_params['linear_go_model']['value_b'] = 0.768941
        chex.assert_trees_all_close(new_params, expected_params)

    def test_value_step_ones_linear_with_single_black_piece(self):
        linear_model = models.get_model('linear')
        states = gojax.decode_state("""
                                    _ _ _
                                    _ B _
                                    _ _ _
                                    """)
        params = linear_model.init(jax.random.PRNGKey(42), states)
        linear_params = params['linear_go_model']
        linear_params['action_w'] = jnp.ones_like(linear_params['action_w'])
        linear_params['value_w'] = jnp.ones_like(linear_params['value_w'])
        linear_params['value_b'] = jnp.ones_like(linear_params['value_b'])

        new_params = train.value_step(linear_model, params, jnp.expand_dims(states, axis=1),
                                      learning_rate=1)

        # Increase value parameters at the index corresponding to the black channel and invalid
        # moves channel.
        # Sigmoid output should be sigmoid(2 + 1) ~ 0.95257. Derivative w.r.t logit is sigmoid -
        # label ~ 047426.
        expected_params = copy.copy(params)
        expected_params['linear_go_model']['value_w'] = params['linear_go_model']['value_w'].at[
            gojax.BLACK_CHANNEL_INDEX, 1, 1].set(1.047426).at[
            gojax.INVALID_CHANNEL_INDEX, 1, 1].set(1.047426)
        expected_params['linear_go_model']['value_b'] = 1.047426

        chex.assert_trees_all_close(new_params, expected_params)

    def test_value_step_ones_linear_with_single_white_piece(self):
        linear_model = models.get_model('linear')
        states = gojax.decode_state("""
                                _ _ _
                                _ W _
                                _ _ _
                                """)
        params = linear_model.init(jax.random.PRNGKey(42), states)
        linear_params = params['linear_go_model']
        linear_params['action_w'] = jnp.ones_like(linear_params['action_w'])
        linear_params['value_w'] = jnp.ones_like(linear_params['value_w'])
        linear_params['value_b'] = jnp.ones_like(linear_params['value_b'])

        new_params = train.value_step(linear_model, params, jnp.expand_dims(states, axis=1),
                                      learning_rate=1)

        # Decrease value parameters at the index corresponding to the black channel and invalid
        # moves channel.
        # Sigmoid output should be sigmoid(2 + 1) ~ 0.95257. Derivative w.r.t logit is sigmoid -
        # label ~ 0.95257.
        expected_params = copy.copy(params)
        expected_params['linear_go_model']['value_w'] = \
            params['linear_go_model']['value_w'].at[gojax.WHITE_CHANNEL_INDEX, 1, 1].set(
                0.047426).at[
                gojax.INVALID_CHANNEL_INDEX, 1, 1].set(0.047426)
        expected_params['linear_go_model']['value_b'] = 0.047426
        chex.assert_trees_all_close(new_params, expected_params, rtol=1e-5)

    if __name__ == '__main__':
        unittest.main()
