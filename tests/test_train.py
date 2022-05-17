"""Tests train.py."""
# pylint: disable=missing-function-docstring,no-self-use,unnecessary-lambda,no-value-for-parameter
import copy
import unittest

import chex
import gojax
import jax
import jax.numpy as jnp
from absl.testing import parameterized

import models
import train


class TrainStepTestCase(chex.TestCase):
    """Tests the value step under the linear model."""

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

        grads = jax.grad(train.compute_k_step_total_loss, argnums=1)(linear_model, params, states,
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
