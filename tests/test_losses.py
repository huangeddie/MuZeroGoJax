"""Tests the loss functions in train_model.py."""
# pylint: disable=missing-function-docstring,no-self-use,unnecessary-lambda,no-value-for-parameter
import unittest
from unittest import mock

import chex
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
    @parameterized.named_parameters(('zeros', [[0, 0]], [[0, 0]], 0.693147), ('ones', [[1, 1]], [[1, 1]], 0.693147),
                                    ('zero_one_one_zero', [[0, 1]], [[1, 0]], 1.04432),
                                    ('zero_one', [[0, 1]], [[0, 1]], 0.582203), # Average of 0.693147 and 0.582203
                                    ('batch_size_two', [[1, 1], [0, 1]], [[1, 1], [0, 1]], 0.637675),
                                    ('three_logits_correct', [[0, 1, 0]], [[0, 1, 0]], 0.975328),
                                    ('three_logits_correct', [[0, 0, 1]], [[0, 0, 1]], 0.975328),
                                    ('cold_temperature', [[0, 0, 1]], [[0, 0, 1]], 0.764459, 0.5),
                                    ('hot_temperature', [[0, 0, 1]], [[0, 0, 1]], 1.099582, 2),
                                    ('scale_logits', [[0, 0, 1]], [[0, 0, 2]], 0.764459), # Same as cold temperature
                                    )
    def test_nd_categorical_cross_entropy(self, action_logits, transition_value_logits, expected_loss, temp=None):
        np.testing.assert_allclose(self.variant(losses.nt_categorical_cross_entropy)(jnp.array(action_logits),
                                                                                     jnp.array(transition_value_logits),
                                                                                     temp), expected_loss, rtol=1e-6)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(('zero_tie', [0], [0.5], 0.693147), ('one_tie', [1], [0.5], 0.813262),
                                    ('neg_one_tie', [-1], [0.5], 0.813262), ('zero_black', [0], [0], 0.693147),
                                    ('zero_white', [0], [1], 0.693147), ('one_black', [1], [0], 1.313262),
                                    ('ones', [1], [1], 0.313262), ('batch_size_two', [0, 1], [1, 0], 1.003204),
                                    # Average of 0.693147 and 1.313262
                                    ('neg_one_black', [-1], [0], 0.313262), ('neg_two_black', [-2], [0], 0.126928), )
    def test_sigmoid_cross_entropy(self, value_logits, labels, expected_loss):
        np.testing.assert_allclose(
            self.variant(losses.nt_sigmoid_cross_entropy)(jnp.array(value_logits), jnp.array(labels)), expected_loss,
            rtol=1e-6)

    @parameterized.named_parameters(('low_loss', [[[1, 0]]], [[[-1, 0]]], 0.582203),
                                    ('mid_loss', [[[0, 0]]], [[[-1, 0]]], 0.693147),
                                    ('high_loss', [[[0, 1]]], [[[-1, 0]]], 1.04432))
    def test_compute_policy_loss_output(self, policy_output, value_output, expected_loss):
        policy_mock_model = mock.Mock(return_value=jnp.array(policy_output))
        value_mock_model = mock.Mock(return_value=jnp.array(value_output))
        params = {}
        step = 0
        transitions = jnp.array([[[0, 0]]])
        nt_embeds = jnp.array([[[0]]])
        np.testing.assert_allclose(
            losses.compute_policy_loss(policy_mock_model, value_mock_model, params, step, transitions, nt_embeds,
                                       temp=1), expected_loss, rtol=1e-6)

    def test_compute_policy_loss_only_policy_has_gradients(self):
        board_size = 2
        value_model = hk.transform(lambda x: models.value.Linear3DValue(board_size, hdim=1)(x))
        policy_model = hk.transform(lambda x: models.policy.Linear3DPolicy(board_size, hdim=1)(x))

        nt_embeds = jnp.reshape(jnp.ones(1), (1, 1, 1, 1, 1))
        params = policy_model.init(jax.random.PRNGKey(42), jnp.zeros((1, 1, 1, 1)))
        transitions = jnp.reshape(jnp.ones(5), (1, 1, 5, 1, 1, 1))
        params.update(value_model.init(jax.random.PRNGKey(42), jnp.zeros((1, 1, 1, 1))))
        step = 0
        grad = jax.grad(losses.compute_policy_loss, argnums=2)(policy_model.apply, value_model.apply, params, step,
                                                               transitions, nt_embeds, temp=1)
        self.assertTrue(grad['linear3_d_policy']['action_w'].astype(bool).all())
        self.assertTrue(~(grad['linear3_d_value']['value_w'].astype(bool).all()))
        self.assertTrue(~grad['linear3_d_value']['value_b'].astype(bool).all())

    def test_compute_value_loss_has_gradients(self):
        board_size = 2
        value_model = hk.transform(lambda x: models.value.Linear3DValue(board_size, hdim=1)(x))

        nt_embeds = jnp.ones((1, 1, 1, 1, 1))
        nt_game_winners = jnp.ones((1, 1, 5, 1, 1, 1))
        params = value_model.init(jax.random.PRNGKey(42), jnp.zeros((1, 1, 1, 1)))
        step = 0
        grad = jax.grad(losses.compute_value_loss, argnums=1)(value_model.apply, params, step, nt_embeds,
                                                              nt_game_winners)
        self.assertTrue(grad['linear3_d_value']['value_w'].astype(bool).all())
        self.assertTrue(grad['linear3_d_value']['value_b'].astype(bool).all())

    def test_compute_value_loss_has_nt_embeds_gradients(self):
        board_size = 2
        value_model = hk.transform(lambda x: models.value.Linear3DValue(board_size, hdim=1)(x))

        nt_embeds = jnp.ones((1, 1, 1, 1, 1))
        nt_game_winners = jnp.ones((1, 1, 5, 1, 1, 1))
        params = value_model.init(jax.random.PRNGKey(42), jnp.zeros((1, 1, 1, 1)))
        step = 0
        grad = jax.grad(losses.compute_value_loss, argnums=3)(value_model.apply, params, step, nt_embeds,
                                                              nt_game_winners)
        self.assertTrue(grad.astype(bool))

    def test_compute_value_loss_step_1_has_nt_embeds_gradients(self):
        board_size = 2
        value_model = hk.transform(lambda x: models.value.Linear3DValue(board_size, hdim=1)(x))

        nt_embeds = jnp.ones((1, 2, 1, 1, 1))
        nt_game_winners = jnp.ones((1, 2, 5, 1, 1, 1))
        params = value_model.init(jax.random.PRNGKey(42), jnp.zeros((1, 1, 1, 1)))
        step = 1
        grad = jax.grad(losses.compute_value_loss, argnums=3)(value_model.apply, params, step, nt_embeds,
                                                              nt_game_winners)
        self.assertTrue(grad.astype(bool).any())

    @parameterized.named_parameters(('low_loss', [[[1]]], [[1]], 0.313262), ('mid_loss', [[[0]]], [[0]], 0.693147),
                                    ('high_loss', [[[-1]]], [[1]], 1.313262))
    def test_compute_value_loss_output(self, value_output, nt_game_winners, expected_loss):
        value_mock_model = mock.Mock(return_value=jnp.array(value_output))
        params = {}
        step = 0
        nt_embeds = jnp.zeros((1, 1, 1))
        nt_game_winners = jnp.array(nt_game_winners)
        np.testing.assert_allclose(
            losses.compute_value_loss(value_mock_model, params, step, nt_embeds, nt_game_winners), expected_loss,
            rtol=1e-6)

    @parameterized.named_parameters(('zero', 1, 1, 0, [[False]]), ('one', 1, 1, 1, [[True]]),
                                    ('zeros', 1, 2, 0, [[False, False]]), ('half', 1, 2, 1, [[True, False]]),
                                    ('full', 1, 2, 2, [[True, True]]), ('b2_zero', 2, 1, 0, [[False], [False]]),
                                    ('b2_one', 2, 1, 1, [[True], [True]]),
                                    ('b2_zeros', 2, 2, 0, [[False, False], [False, False]]),
                                    ('b2_half', 2, 2, 1, [[True, False], [True, False]]),
                                    ('b2_full', 2, 2, 2, [[True, True], [True, True]]), )
    def test_k_steps_mask(self, batch_size, total_steps, k, expected_output):
        """Tests the make_nt_mask based on inputs and expected output."""
        np.testing.assert_array_equal(losses.make_nt_mask(batch_size, total_steps, k), expected_output)

    def test_compute_0_step_total_loss_has_gradients_except_for_transitions(self):
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

        # Check all transition weights are 0.
        self.assertTrue((~grad['linear3_d_transition']['transition_b'].astype(bool)).all())
        self.assertTrue((~grad['linear3_d_transition']['transition_w'].astype(bool)).all())
        # Check everything else is non-zero.
        del grad['linear3_d_transition']['transition_b']
        del grad['linear3_d_transition']['transition_w']
        self.assertTrue(self.tree_leaves_all_non_zero(grad), aux)

    def test_compute_1_step_total_loss_has_gradients_with_some_transitions(self):
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --hdim=2 --embed_model=linear --value_model=linear '
                   '--policy_model=linear --transition_model=linear'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        trajectories = jnp.ones((1, 2, 6, 3, 3), dtype=bool)
        grad_loss_fn = jax.grad(losses.compute_k_step_total_loss, argnums=1, has_aux=True)
        grad, aux = grad_loss_fn(go_model, params, trajectories, k=2)

        self.assertTrue(grad['linear3_d_transition']['transition_b'].astype(bool).any())
        self.assertTrue(grad['linear3_d_transition']['transition_w'].astype(bool).any())
        # Check everything else is all non-zero.
        del grad['linear3_d_transition']['transition_b']
        del grad['linear3_d_transition']['transition_w']
        self.assertTrue(self.tree_leaves_all_non_zero(grad), aux)

    if __name__ == '__main__':
        unittest.main()
