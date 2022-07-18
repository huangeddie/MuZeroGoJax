"""Tests the loss functions in train_model.py."""
# pylint: disable=missing-function-docstring,no-self-use,unnecessary-lambda,no-value-for-parameter
import unittest
from unittest import mock

import chex
import gojax
import haiku as hk
import jax.random
import numpy as np
from absl.testing import parameterized
from jax import numpy as jnp

from muzero_gojax import losses
from muzero_gojax import main
from muzero_gojax import models


class LossesTestCase(chex.TestCase):
    """Test policy loss under various inputs"""

    def tree_leaves_all_non_zero(self, tree_dict):
        for key, val in tree_dict.items():
            if isinstance(val, dict):
                if not self.tree_leaves_all_non_zero(val):
                    print('Tree branch zero: ', key)
                    return False
            else:
                if not jnp.alltrue(val.astype(bool)):
                    print('Tree leaf zero: ', key, val.shape, val)
                    return False
        return True

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
    def test_nd_categorical_cross_entropy(self, action_logits, transition_value_logits,
                                          expected_loss, temp=None):
        np.testing.assert_allclose(
            self.variant(losses.nd_categorical_cross_entropy)(jnp.array(action_logits),
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
        np.testing.assert_allclose(
            self.variant(losses.sigmoid_cross_entropy)(jnp.array(value_logits), jnp.array(labels)),
            expected_loss, rtol=1e-6)

    @parameterized.named_parameters(('low_loss', [[[1, 0]]], [[[1, 0]]], 0.582203),
                                    ('mid_loss', [[[0, 0]]], [[[1, 0]]], 0.693147),
                                    ('high_loss', [[[0, 1]]], [[[1, 0]]], 1.04432))
    def test_compute_policy_loss_output(self, policy_output, value_output, expected_loss):
        policy_mock_model = mock.Mock(return_value=jnp.array(policy_output))
        value_mock_model = mock.Mock(return_value=jnp.array(value_output))
        params = {}
        step = 0
        transitions = jnp.array([[[0, 0]]])
        nt_embeds = jnp.array([[[0]]])
        np.testing.assert_allclose(
            losses.compute_policy_loss(policy_mock_model, value_mock_model, params, step,
                                       transitions, nt_embeds), expected_loss, rtol=1e-6)

    def test_compute_policy_loss_only_policy_has_gradients(self):
        board_size = 2
        value_model = hk.transform(lambda x: models.value.Linear3DValue(board_size, hdim=1)(x))
        policy_model = hk.transform(lambda x: models.policy.Linear3DPolicy(board_size, hdim=1)(x))

        nt_embeds = jnp.reshape(jnp.ones(1), (1, 1, 1, 1, 1))
        params = policy_model.init(jax.random.PRNGKey(42), jnp.zeros((1, 1, 1, 1)))
        transitions = jnp.reshape(jnp.ones(5), (1, 1, 5, 1, 1, 1))
        params.update(value_model.init(jax.random.PRNGKey(42), jnp.zeros((1, 1, 1, 1))))
        step = 0
        grad = jax.grad(losses.compute_policy_loss, argnums=2)(policy_model.apply,
                                                               value_model.apply, params, step,
                                                               transitions, nt_embeds)
        self.assertTrue(jnp.alltrue(grad['linear3_d_policy']['action_w'].astype(bool)))
        self.assertTrue(jnp.alltrue(~grad['linear3_d_value']['value_w'].astype(bool)))
        self.assertTrue(jnp.alltrue(~grad['linear3_d_value']['value_b'].astype(bool)))

    def test_compute_value_loss_has_gradients(self):
        board_size = 2
        value_model = hk.transform(lambda x: models.value.Linear3DValue(board_size, hdim=1)(x))

        nt_embeds = jnp.ones((1, 1, 1, 1, 1))
        nt_game_winners = jnp.ones((1, 1, 5, 1, 1, 1))
        params = value_model.init(jax.random.PRNGKey(42), jnp.zeros((1, 1, 1, 1)))
        step = 0
        grad = jax.grad(losses.compute_value_loss, argnums=1)(value_model.apply, params, step,
                                                              nt_embeds, nt_game_winners)
        self.assertTrue(jnp.alltrue(grad['linear3_d_value']['value_w'].astype(bool)))
        self.assertTrue(jnp.alltrue(grad['linear3_d_value']['value_b'].astype(bool)))

    def test_compute_value_loss_has_nt_embeds_gradients(self):
        board_size = 2
        value_model = hk.transform(lambda x: models.value.Linear3DValue(board_size, hdim=1)(x))

        nt_embeds = jnp.ones((1, 1, 1, 1, 1))
        nt_game_winners = jnp.ones((1, 1, 5, 1, 1, 1))
        params = value_model.init(jax.random.PRNGKey(42), jnp.zeros((1, 1, 1, 1)))
        step = 0
        grad = jax.grad(losses.compute_value_loss, argnums=3)(value_model.apply, params, step,
                                                              nt_embeds, nt_game_winners)
        self.assertTrue(grad.astype(bool))

    @parameterized.named_parameters(('low_loss', [[[1]]], [[1]], 0.313262),
                                    ('mid_loss', [[[0]]], [[0]], 0.693147),
                                    ('high_loss', [[[-1]]], [[1]], 1.313262))
    def test_compute_value_loss_output(self, value_output, nt_game_winners, expected_loss):
        value_mock_model = mock.Mock(return_value=jnp.array(value_output))
        params = {}
        step = 0
        nt_embeds = jnp.zeros((1, 1, 1))
        nt_game_winners = jnp.array(nt_game_winners)
        np.testing.assert_allclose(
            losses.compute_value_loss(value_mock_model, params, step, nt_embeds, nt_game_winners),
            expected_loss, rtol=1e-6)

    @parameterized.named_parameters(('zero', 1, 1, 0, [[False]]), ('one', 1, 1, 1, [[True]]),
                                    ('zeros', 1, 2, 0, [[False, False]]),
                                    ('half', 1, 2, 1, [[True, False]]),
                                    ('full', 1, 2, 2, [[True, True]]),
                                    ('b2_zero', 2, 1, 0, [[False], [False]]),
                                    ('b2_one', 2, 1, 1, [[True], [True]]),
                                    ('b2_zeros', 2, 2, 0, [[False, False], [False, False]]),
                                    ('b2_half', 2, 2, 1, [[True, False], [True, False]]),
                                    ('b2_full', 2, 2, 2, [[True, True], [True, True]]), )
    def test_k_steps_mask(self, batch_size, total_steps, k, expected_output):
        """Tests the make_first_k_steps_mask based on inputs and expected output."""
        np.testing.assert_array_equal(losses.make_first_k_steps_mask(batch_size, total_steps, k),
                                      expected_output)

    @parameterized.named_parameters(
        {'testcase_name': 'zero_output_tie_game', 'embed_output': [[[0.]]], 'value_output': [[0.]],
         'second_value_output': [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
         'policy_output': jnp.zeros((1, 10)), 'transition_output': jnp.zeros((1, 10, 1)),
         'game_winners': [[0]], 'expected_value_loss': 0.6931471806,
         'expected_policy_loss': 2.302585},
        {'testcase_name': 'zero_output_black_wins', 'embed_output': [[[0.]]],
         'value_output': [[0.]], 'second_value_output': [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
         'policy_output': jnp.zeros((1, 10)), 'transition_output': jnp.zeros((1, 10, 1)),
         'game_winners': [[1]], 'expected_value_loss': 0.6931471806,
         'expected_policy_loss': 2.302585},
        {'testcase_name': 'zero_output_white_wins', 'embed_output': [[[0.]]],
         'value_output': [[0.]], 'second_value_output': [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
         'policy_output': jnp.zeros((1, 10)), 'transition_output': jnp.zeros((1, 10, 1)),
         'game_winners': [[1]], 'expected_value_loss': 0.6931471806,
         'expected_policy_loss': 2.302585},
        {'testcase_name': 'one_output_tie_game', 'embed_output': [[[1.]]], 'value_output': [[1.]],
         'second_value_output': [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]],
         'policy_output': jnp.ones((1, 10)), 'transition_output': jnp.ones((1, 10, 1)),
         'game_winners': [[0]], 'expected_value_loss': 0.81326175,
         'expected_policy_loss': 2.302585},
        {'testcase_name': 'one_output_black_wins', 'embed_output': [[[1.]]], 'value_output': [[1.]],
         'second_value_output': [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]],
         'policy_output': jnp.ones((1, 10)), 'transition_output': jnp.ones((1, 10, 1)),
         'game_winners': [[1]], 'expected_value_loss': 0.3132617, 'expected_policy_loss': 2.302585},
        {'testcase_name': 'one_output_white_wins', 'embed_output': [[[1.]]], 'value_output': [[1.]],
         'second_value_output': [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]],
         'policy_output': jnp.ones((1, 10)), 'transition_output': jnp.ones((1, 10, 1)),
         'game_winners': [[-1]], 'expected_value_loss': 1.3132617,
         'expected_policy_loss': 2.302585}, )
    def test_compute_k_step_losses_outputs(self, embed_output, value_output, second_value_output,
                                           policy_output, transition_output, game_winners,
                                           expected_value_loss, expected_policy_loss):
        # pylint: disable=too-many-arguments
        # Build the mock model.
        mock_model = mock.Mock()
        embed_mock_model = mock.Mock(return_value=jnp.array(embed_output))
        value_mock_model = mock.Mock(
            side_effect=[jnp.array(value_output), jnp.array(second_value_output)] * 2)
        policy_mock_model = mock.Mock(return_value=jnp.array(policy_output))
        transition_mock_model = mock.Mock(return_value=jnp.array(transition_output))
        mock_model.apply = (
            embed_mock_model, value_mock_model, policy_mock_model, transition_mock_model,)

        # Execute the loss function.
        loss_dict = losses.compute_k_step_losses(mock_model, {}, jnp.expand_dims(
            gojax.new_states(board_size=3, batch_size=1), 0), jnp.array([[-1]]),
                                                 jnp.array(game_winners))
        # Test.
        self.assertLen(loss_dict, 2)
        self.assertIn('cum_val_loss', loss_dict)
        self.assertIn('cum_policy_loss', loss_dict)
        self.assertAlmostEqual(loss_dict['cum_val_loss'], expected_value_loss)
        self.assertAlmostEqual(loss_dict['cum_policy_loss'], expected_policy_loss)
        self.assertEqual(embed_mock_model.call_count, 1)
        # Compiles the body_fun in the for loop so the mocks below are called twice the normal
        # amount.
        self.assertEqual(value_mock_model.call_count, 4)
        self.assertEqual(policy_mock_model.call_count, 2)
        self.assertEqual(transition_mock_model.call_count, 2)

    def test_compute_k_step_total_loss_has_gradients(self):
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --hdim=2 --embed_model=linear --value_model=linear '
                   '--policy_model=linear --transition_model=linear'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        trajectories = jnp.ones((1, 1, 6, 3, 3), dtype=bool)
        actions = jnp.ones((1, 1), dtype=int)
        game_winners = jnp.ones((1, 1), dtype=int)
        grad_loss_fn = jax.grad(losses.compute_k_step_total_loss, argnums=1, has_aux=True)
        grad, aux = grad_loss_fn(go_model, params, trajectories, actions, game_winners)
        self.assertTrue(self.tree_leaves_all_non_zero(grad), aux)

    if __name__ == '__main__':
        unittest.main()
