"""Tests train.py."""
# pylint: disable=missing-function-docstring,no-self-use,unnecessary-lambda
import unittest

import chex
import gojax
import jax
import jax.numpy as jnp

import models
import train


class TrainTestCase(unittest.TestCase):
    """Tests train.py."""

    def test_value_loss_fn_zeros_linear_with_tie_labels(self):
        linear_model = models.get_model('linear')
        states = jnp.ones_like(gojax.new_states(batch_size=1, board_size=3))
        params = linear_model.init(jax.random.PRNGKey(42), states)
        linear_params = params['linear_go_model']
        linear_params['value_w'] = jnp.zeros_like(linear_params['value_w'])
        linear_params['value_b'] = jnp.zeros_like(linear_params['value_b'])
        self.assertAlmostEqual(
            train.value_loss_fn(linear_model, params, states, jnp.zeros(len(states))),
            0.6931471806)

    def test_value_loss_fn_zeros_linear_with_black_won_labels(self):
        linear_model = models.get_model('linear')
        states = jnp.ones_like(gojax.new_states(batch_size=1, board_size=3))
        params = linear_model.init(jax.random.PRNGKey(42), states)
        linear_params = params['linear_go_model']
        linear_params['value_w'] = jnp.zeros_like(linear_params['value_w'])
        linear_params['value_b'] = jnp.zeros_like(linear_params['value_b'])
        self.assertAlmostEqual(
            train.value_loss_fn(linear_model, params, states, jnp.ones(len(states))),
            0.6931471806)

    def test_value_loss_fn_zeros_linear_with_white_won_labels(self):
        linear_model = models.get_model('linear')
        states = jnp.ones_like(gojax.new_states(batch_size=1, board_size=3))
        params = linear_model.init(jax.random.PRNGKey(42), states)
        linear_params = params['linear_go_model']
        linear_params['value_w'] = jnp.zeros_like(linear_params['value_w'])
        linear_params['value_b'] = jnp.zeros_like(linear_params['value_b'])
        self.assertAlmostEqual(
            train.value_loss_fn(linear_model, params, states, -jnp.ones(len(states))),
            0.6931471806)

    def test_value_loss_fn_ones_linear_with_tie_labels(self):
        linear_model = models.get_model('linear')
        states = jnp.ones_like(gojax.new_states(batch_size=1, board_size=3))
        params = linear_model.init(jax.random.PRNGKey(42), states)
        linear_params = params['linear_go_model']
        linear_params['value_w'] = jnp.ones_like(linear_params['value_w'])
        linear_params['value_b'] = jnp.ones_like(linear_params['value_b'])

        # High loss
        self.assertAlmostEqual(
            train.value_loss_fn(linear_model, params, states, jnp.zeros(len(states))),
            27.5)

    def test_value_loss_fn_ones_linear_with_black_won_labels(self):
        linear_model = models.get_model('linear')
        states = jnp.ones_like(gojax.new_states(batch_size=1, board_size=3))
        params = linear_model.init(jax.random.PRNGKey(42), states)
        linear_params = params['linear_go_model']
        linear_params['value_w'] = jnp.ones_like(linear_params['value_w'])
        linear_params['value_b'] = jnp.ones_like(linear_params['value_b'])

        # Low loss
        self.assertAlmostEqual(
            train.value_loss_fn(linear_model, params, states, jnp.ones(len(states))),
            1.2995815e-24)

    def test_value_loss_fn_ones_linear_with_white_won_labels(self):
        linear_model = models.get_model('linear')
        states = jnp.ones_like(gojax.new_states(batch_size=1, board_size=3))
        params = linear_model.init(jax.random.PRNGKey(42), states)
        linear_params = params['linear_go_model']
        linear_params['value_w'] = jnp.ones_like(linear_params['value_w'])
        linear_params['value_b'] = jnp.ones_like(linear_params['value_b'])

        # Very high loss
        self.assertAlmostEqual(
            train.value_loss_fn(linear_model, params, states, -jnp.ones(len(states))),
            55)

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

        # Decrease value parameters towards 0.
        chex.assert_trees_all_close(grads, {
            'linear_go_model': {'action_w': jnp.zeros_like(params['linear_go_model']['action_w']),
                                'value_w': jnp.full_like(
                                    jnp.zeros_like(params['linear_go_model']['value_w']),
                                    fill_value=0.5),
                                'value_b': 0.5, }})

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
        chex.assert_trees_all_close(new_params, {
            'linear_go_model': {'action_w': params['linear_go_model']['action_w'],
                                'value_w': params['linear_go_model']['value_w'],
                                'value_b': 0.768941, }})

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
        chex.assert_trees_all_close(new_params, {
            'linear_go_model': {'action_w': params['linear_go_model']['action_w'],
                                'value_w':
                                    params['linear_go_model']['value_w'].at[4].set(1.047426).at[
                                        31].set(1.047426),
                                'value_b': 1.047426, }})

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
        chex.assert_trees_all_close(new_params, {
            'linear_go_model': {'action_w': params['linear_go_model']['action_w'],
                                'value_w':
                                    params['linear_go_model']['value_w'].at[13].set(0.047426).at[
                                        31].set(0.047426),
                                'value_b': 0.047426, }}, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
