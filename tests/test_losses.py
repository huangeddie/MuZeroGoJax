"""Tests the loss functions in train_model.py."""

import unittest

import chex
import gojax
import jax.random
import numpy as np
from absl.testing import parameterized
from jax import numpy as jnp

from muzero_gojax import losses
from muzero_gojax import main
from muzero_gojax import models


def test_compute_embed_loss_with_full_mask():
    transitions = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 2))
    expected_transitions = jax.random.normal(jax.random.PRNGKey(69), (2, 2, 2))
    np.testing.assert_allclose(losses.compute_trans_loss(expected_transitions, transitions,
                                                         losses.make_prefix_nt_mask(2, 2, 2)),
                               7.4375)


def test_compute_embed_loss_with_half_mask():
    transitions = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 2))
    expected_transitions = jax.random.normal(jax.random.PRNGKey(69), (2, 2, 2))
    np.testing.assert_allclose(losses.compute_trans_loss(expected_transitions, transitions,
                                                         losses.make_prefix_nt_mask(2, 2, 1)),
                               14.1875)


class LossesTestCase(chex.TestCase):
    """Test policy loss under various inputs"""

    def tree_leaves_all_non_zero(self, tree_dict):
        """Returns whether the all leaves are non-zero."""
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
                                    ('batch_size_two', [[1, 1], [0, 1]], [[1, 1], [0, 1]],
                                     0.637675),
                                    ('three_logits_correct', [[0, 1, 0]], [[0, 1, 0]], 0.975328),
                                    ('three_logits_correct', [[0, 0, 1]], [[0, 0, 1]], 0.975328),
                                    ('cold_temperature', [[0, 0, 1]], [[0, 0, 1]], 0.764459, 0.5),
                                    ('hot_temperature', [[0, 0, 1]], [[0, 0, 1]], 1.099582, 2),
                                    ('scale_logits', [[0, 0, 1]], [[0, 0, 2]], 0.764459),
                                    # Same as cold temperature
                                    )
    def test_nt_categorical_cross_entropy(self, action_logits, transition_value_logits,
                                          expected_loss, temp=None):
        """Tests the nt_categorical_cross_entropy."""
        np.testing.assert_allclose(
            self.variant(losses.nt_categorical_cross_entropy)(jnp.array(action_logits),
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
        """Tests the nt_sigmoid_cross_entropy."""
        np.testing.assert_allclose(
            self.variant(losses.nt_sigmoid_cross_entropy)(jnp.array(value_logits),
                                                          jnp.array(labels)), expected_loss,
            rtol=1e-6)

    @parameterized.named_parameters(('low_loss', [[[1, 0]]], [[[-1, 0]]], 0.582203),
                                    ('mid_loss', [[[0, 0]]], [[[-1, 0]]], 0.693147),
                                    ('high_loss', [[[0, 1]]], [[[-1, 0]]], 1.04432))
    def test_compute_policy_loss_output(self, policy_output, value_output, expected_loss):
        """Tests the compute_policy_loss."""
        np.testing.assert_allclose(
            losses.compute_policy_loss(jnp.array(policy_output), jnp.array(value_output),
                                       hypo_step=0, temp=1), expected_loss, rtol=1e-6)

    def test_compute_value_loss_low_value(self):
        """Tests gradient of compute_value_loss w.r.t to params."""
        self.assertEqual(losses.compute_value_loss(value_logits=-jnp.ones((1, 1)),
                                                   nt_game_winners=-jnp.ones((1, 1)), hypo_step=0),
                         0.3132617)

    def test_compute_value_loss_high_value(self):
        """Tests gradient of compute_value_loss w.r.t to params."""
        self.assertEqual(losses.compute_value_loss(value_logits=-jnp.ones((1, 1)),
                                                   nt_game_winners=jnp.ones((1, 1)), hypo_step=0),
                         1.3132617)

    @parameterized.named_parameters(('zero', 1, 1, 0, [[False]]), ('one', 1, 1, 1, [[True]]),
                                    ('zeros', 1, 2, 0, [[False, False]]),
                                    ('half', 1, 2, 1, [[True, False]]),
                                    ('full', 1, 2, 2, [[True, True]]),
                                    ('b2_zero', 2, 1, 0, [[False], [False]]),
                                    ('b2_one', 2, 1, 1, [[True], [True]]),
                                    ('b2_zeros', 2, 2, 0, [[False, False], [False, False]]),
                                    ('b2_half', 2, 2, 1, [[True, False], [True, False]]),
                                    ('b2_full', 2, 2, 2, [[True, True], [True, True]]), )
    def test_prefix_k_steps_mask(self, batch_size, total_steps, k, expected_output):
        """Tests the make_prefix_nt_mask based on inputs and expected output."""
        np.testing.assert_array_equal(losses.make_prefix_nt_mask(batch_size, total_steps, k),
                                      expected_output)

    @parameterized.named_parameters(('zero', 1, 1, 0, [[False]]), ('one', 1, 1, 1, [[True]]),
                                    ('zeros', 1, 2, 0, [[False, False]]),
                                    ('half', 1, 2, 1, [[False, True]]),
                                    ('full', 1, 2, 2, [[True, True]]),
                                    ('b2_zero', 2, 1, 0, [[False], [False]]),
                                    ('b2_one', 2, 1, 1, [[True], [True]]),
                                    ('b2_zeros', 2, 2, 0, [[False, False], [False, False]]),
                                    ('b2_half', 2, 2, 1, [[False, True], [False, True]]),
                                    ('b2_full', 2, 2, 2, [[True, True], [True, True]]), )
    def test_suffix_k_steps_mask(self, batch_size, total_steps, k, expected_output):
        """Tests the make_prefix_nt_mask based on inputs and expected output."""
        np.testing.assert_array_equal(losses.make_suffix_nt_mask(batch_size, total_steps, k),
                                      expected_output)

    def test_compute_0_step_total_loss_has_gradients_except_for_transitions(self):
        """Tests all parameters except for transitions have grads with compute_0_step_total_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --hdim=2 --embed_model=linear --value_model=linear '
                   '--policy_model=linear --transition_model=linear --hypo_steps=1'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        trajectories = jnp.ones((1, 1, 6, 3, 3), dtype=bool)
        grad_loss_fn = jax.grad(losses.compute_k_step_total_loss, argnums=2, has_aux=True)
        grad, aux = grad_loss_fn(main.FLAGS, go_model, params, trajectories)

        # Check all transition weights are 0.
        self.assertTrue((~grad['linear3_d_transition']['transition_b'].astype(bool)).all())
        self.assertTrue((~grad['linear3_d_transition']['transition_w'].astype(bool)).all())
        # Check everything else is non-zero.
        del grad['linear3_d_transition']['transition_b']
        del grad['linear3_d_transition']['transition_w']
        self.assertTrue(self.tree_leaves_all_non_zero(grad), aux)

    def test_compute_1_step_total_loss_has_gradients_with_some_transitions(self):
        """Tests some transitions params have grads with compute_1_step_total_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --hdim=2 --embed_model=linear --value_model=linear '
                   '--policy_model=linear --transition_model=linear --hypo_steps=2'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        trajectories = jnp.ones((1, 2, 6, 3, 3), dtype=bool)
        grad_loss_fn = jax.grad(losses.compute_k_step_total_loss, argnums=2, has_aux=True)
        grad, aux = grad_loss_fn(main.FLAGS, go_model, params, trajectories)

        self.assertTrue(grad['linear3_d_transition']['transition_b'].astype(bool).any())
        self.assertTrue(grad['linear3_d_transition']['transition_w'].astype(bool).any())
        # Check everything else is all non-zero.
        del grad['linear3_d_transition']['transition_b']
        del grad['linear3_d_transition']['transition_w']
        self.assertTrue(self.tree_leaves_all_non_zero(grad), aux)

    def test_update_0_step_loss_black_perspective_zero_embed_loss(self):
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --embed_model=black_perspective --value_model=linear '
                   '--policy_model=linear --transition_model=black_perspective'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        black_embeds = gojax.decode_states("""
                                            _ _ _
                                            _ _ _
                                            _ _ _

                                            _ _ _
                                            _ W _
                                            _ _ _
                                            """)
        metrics_data = losses.update_k_step_losses(go_model, params, temp=1, i=0, data={
            'nt_embeds': jnp.reshape(black_embeds, (1, 2, 6, 3, 3)),
            'nt_actions': jnp.array([[4, 4]]), 'nt_game_winners': jnp.array([[1, -1]]),
            'cum_val_loss': 0, 'cum_policy_loss': 0, 'cum_trans_loss': 0,
        })
        self.assertIn('cum_trans_loss', metrics_data)
        self.assertEqual(metrics_data['cum_trans_loss'], 0)

    def test_update_0_step_loss_black_perspective_zero_trans_loss_length_3(self):
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --embed_model=black_perspective --value_model=linear '
                   '--policy_model=linear --transition_model=black_perspective'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        black_embeds = gojax.decode_states("""
                                            _ _ _
                                            _ _ _
                                            _ _ _

                                            _ _ _
                                            _ _ _
                                            _ _ W
                                            
                                            _ _ _
                                            _ _ _
                                            W _ B
                                            """)
        metrics_data = losses.update_k_step_losses(go_model, params, temp=1, i=0, data={
            'nt_embeds': jnp.reshape(black_embeds, (1, 3, 6, 3, 3)),
            'nt_actions': jnp.array([[8, 6, 6]]), 'nt_game_winners': jnp.array([[0, 0, 0]]),
            'cum_val_loss': 0, 'cum_policy_loss': 0, 'cum_trans_loss': 0,
        })
        self.assertIn('cum_trans_loss', metrics_data)
        self.assertEqual(metrics_data['cum_trans_loss'], 0)

    def test_update_1_step_loss_black_perspective_zero_embed_loss(self):
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --embed_model=black_perspective --value_model=linear '
                   '--policy_model=linear --transition_model=black_perspective'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        black_embeds = gojax.decode_states("""
                                            _ _ _
                                            _ _ _
                                            _ _ _

                                            _ _ _
                                            _ W _
                                            _ _ _
                                            TURN=B
                                            """)
        metrics_data = losses.update_k_step_losses(go_model, params, temp=1, i=1, data={
            'nt_embeds': jnp.reshape(black_embeds, (1, 2, 6, 3, 3)),
            'nt_actions': jnp.array([[4, 4]]), 'nt_game_winners': jnp.array([[1, -1]]),
            'cum_val_loss': 0, 'cum_policy_loss': 0, 'cum_trans_loss': 0,
        })
        self.assertIn('cum_trans_loss', metrics_data)
        self.assertEqual(metrics_data['cum_trans_loss'], 0)

    def test_update_0_step_loss_black_perspective_zero_embed_loss_batches(self):
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --embed_model=black_perspective --value_model=linear '
                   '--policy_model=linear --transition_model=black_perspective'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        black_embeds = gojax.decode_states("""
                                            _ _ _
                                            _ _ _
                                            _ _ _

                                            _ _ _
                                            _ W _
                                            _ _ _
                                            TURN=B
                                            
                                            _ B _
                                            B W _
                                            _ B _
                                            
                                            _ W _
                                            W X W
                                            _ W _
                                            TURN=B
                                            """)
        metrics_data = losses.update_k_step_losses(go_model, params, temp=1, i=0, data={
            'nt_embeds': jnp.reshape(black_embeds, (2, 2, 6, 3, 3)),
            'nt_actions': jnp.array([[4, 4], [5, 5]]),
            'nt_game_winners': jnp.array([[1, -1], [1, -1]]), 'cum_val_loss': 0,
            'cum_policy_loss': 0, 'cum_trans_loss': 0,
        })
        self.assertIn('cum_trans_loss', metrics_data)
        self.assertEqual(metrics_data['cum_trans_loss'], 0)

    def test_compute_1_step_losses_black_perspective_hard(self):
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --embed_model=black_perspective --value_model=linear '
                   '--policy_model=linear --transition_model=black_perspective'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        trajectories = gojax.decode_states("""
                                            B _ W 
                                            W _ _ 
                                            _ W W 
                                            TURN=W
                                            
                                            B _ W 
                                            W W _ 
                                            _ W W 
                                            
                                            B B _ 
                                            B X B 
                                            X B X 
                                            TURN=W
                                            
                                            B B _ 
                                            B _ B 
                                            _ B _ 
                                            PASS=T
                                            """)
        trajectories = jnp.reshape(trajectories, (2, 2, 6, 3, 3))
        metrics_data = losses.compute_k_step_losses(go_model, params, trajectories, k=1)
        self.assertIn('cum_trans_loss', metrics_data)
        self.assertEqual(metrics_data['cum_trans_loss'], 0)

    def test_compute_2_step_losses_black_perspective_length_2(self):
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --embed_model=black_perspective --value_model=linear '
                   '--policy_model=linear --transition_model=black_perspective'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        trajectories = gojax.decode_states("""
                                            _ _ _
                                            _ _ _
                                            _ _ _

                                            _ _ _
                                            _ B _
                                            _ _ _
                                            TURN=W
                                            """)
        trajectories = jnp.reshape(trajectories, (1, 2, 6, 3, 3))
        metrics_data = losses.compute_k_step_losses(go_model, params, trajectories, k=2)
        self.assertIn('cum_trans_loss', metrics_data)
        self.assertEqual(metrics_data['cum_trans_loss'], 0)

    def test_compute_2_step_losses_black_perspective_batches(self):
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --embed_model=black_perspective --value_model=linear '
                   '--policy_model=linear --transition_model=black_perspective'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        trajectories = gojax.decode_states("""
                                            _ _ _
                                            _ _ _
                                            _ _ _

                                            _ _ _
                                            _ B _
                                            _ _ _
                                            TURN=W
                                            
                                            _ B _
                                            B W _
                                            _ B _
                                            
                                            _ B _
                                            B X B
                                            _ B _
                                            TURN=W
                                            """)
        trajectories = jnp.reshape(trajectories, (2, 2, 6, 3, 3))
        metrics_data = losses.compute_k_step_losses(go_model, params, trajectories, k=2)
        self.assertIn('cum_trans_loss', metrics_data)
        self.assertEqual(metrics_data['cum_trans_loss'], 0)

    if __name__ == '__main__':
        unittest.main()
