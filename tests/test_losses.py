"""Tests the loss functions in train_model.py."""
# pylint: disable=missing-function-docstring,no-self-use,no-value-for-parameter

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


def test_get_flat_trans_logits_with_fixed_input_no_embed_gradient_through_params():
    main.FLAGS.unparse_flags()
    main.FLAGS('foo --board_size=3 --embed_model=linear_conv --value_model=linear '
               '--policy_model=linear --transition_model=linear_conv'.split())
    go_model = models.make_model(main.FLAGS)
    states = jnp.ones_like(gojax.new_states(board_size=3, batch_size=1))
    params = go_model.init(jax.random.PRNGKey(42), states)
    embed_model, _, _, transition_model = go_model.apply
    nt_embed = jnp.reshape(embed_model(params, None, states), (1, 1, main.FLAGS.embed_dim, 3, 3))
    grads = jax.grad(lambda transition_model_, params_, nt_embed_: jnp.sum(
        losses.get_flat_trans_logits_with_fixed_input(transition_model_, params_, nt_embed_)),
                     argnums=1)(transition_model, params, nt_embed)

    np.testing.assert_array_equal(grads['linear_conv_embed/~/conv2_d']['b'],
                                  jnp.zeros_like(grads['linear_conv_embed/~/conv2_d']['b']))
    np.testing.assert_array_equal(grads['linear_conv_embed/~/conv2_d']['w'],
                                  jnp.zeros_like(grads['linear_conv_embed/~/conv2_d']['w']))
    np.testing.assert_array_equal(grads['linear_conv_transition/~/conv2_d']['b'].astype(bool),
                                  jnp.ones_like(
                                      grads['linear_conv_transition/~/conv2_d']['b'].astype(bool)))
    np.testing.assert_array_equal(grads['linear_conv_transition/~/conv2_d']['w'].astype(bool),
                                  jnp.ones_like(
                                      grads['linear_conv_transition/~/conv2_d']['w'].astype(bool)))


def test_get_flat_trans_logits_with_fixed_input_no_embed_gradient_through_embeds():
    main.FLAGS.unparse_flags()
    main.FLAGS('foo --board_size=3 --embed_model=linear_conv --value_model=linear '
               '--policy_model=linear --transition_model=linear_conv'.split())
    go_model = models.make_model(main.FLAGS)
    states = jnp.ones_like(gojax.new_states(board_size=3, batch_size=1))
    params = go_model.init(jax.random.PRNGKey(42), states)
    embed_model, _, _, transition_model = go_model.apply
    nt_embed = jnp.reshape(embed_model(params, None, states), (1, 1, main.FLAGS.embed_dim, 3, 3))
    grads = jax.grad(lambda transition_model_, params_, nt_embed_: jnp.sum(
        losses.get_flat_trans_logits_with_fixed_input(transition_model_, params_, nt_embed_)),
                     argnums=2)(transition_model, params, nt_embed)

    np.testing.assert_array_equal(grads, jnp.zeros_like(grads))


class LossesTestCase(chex.TestCase):
    """Test policy loss under various inputs"""

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

    def test_kl_div_trans_loss_with_full_mask(self):
        transitions = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 2))
        expected_transitions = jax.random.normal(jax.random.PRNGKey(69), (2, 2, 2))
        np.testing.assert_allclose(losses.kl_div_trans_loss(transitions, expected_transitions,
                                                            losses.make_prefix_nt_mask(2, 2, 2)),
                                   0.464844, atol=1e-5)

    def test_kl_div_trans_loss_with_half_mask(self):
        transitions = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 2))
        expected_transitions = jax.random.normal(jax.random.PRNGKey(69), (2, 2, 2))
        np.testing.assert_allclose(losses.kl_div_trans_loss(transitions, expected_transitions,
                                                            losses.make_prefix_nt_mask(2, 2, 1)),
                                   0.777344, atol=1e-5)

    def test_mse_trans_loss_with_full_mask(self):
        transitions = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 2))
        expected_transitions = jax.random.normal(jax.random.PRNGKey(69), (2, 2, 2))
        np.testing.assert_allclose(losses.mse_trans_loss(transitions, expected_transitions,
                                                         losses.make_prefix_nt_mask(2, 2, 2)),
                                   3.718629, atol=1e-5)

    def test_mse_trans_loss_with_half_mask(self):
        transitions = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 2))
        expected_transitions = jax.random.normal(jax.random.PRNGKey(69), (2, 2, 2))
        np.testing.assert_allclose(losses.mse_trans_loss(transitions, expected_transitions,
                                                         losses.make_prefix_nt_mask(2, 2, 1)),
                                   7.082739, atol=1e-5)

    def test_bce_trans_loss_with_full_mask(self):
        transitions = jax.random.uniform(jax.random.PRNGKey(42), (2, 2, 2))
        expected_transitions = jax.random.bernoulli(jax.random.PRNGKey(69), shape=(2, 2, 2)).astype(
            'bfloat16')
        np.testing.assert_allclose(losses.bce_trans_loss(transitions, expected_transitions,
                                                         losses.make_prefix_nt_mask(2, 2, 2)),
                                   3.399168, atol=1e-5)

    def test_bce_trans_loss_with_half_mask(self):
        transitions = jax.random.uniform(jax.random.PRNGKey(42), (2, 2, 2))
        expected_transitions = jax.random.bernoulli(jax.random.PRNGKey(69), shape=(2, 2, 2)).astype(
            'bfloat16')
        np.testing.assert_allclose(losses.bce_trans_loss(transitions, expected_transitions,
                                                         losses.make_prefix_nt_mask(2, 2, 1)),
                                   0.947677, atol=1e-5)

    def test_bce_trans_loss_with_extreme_values(self):
        transitions = jnp.array([[[1]]], dtype='bfloat16')
        expected_transitions = jax.random.bernoulli(jax.random.PRNGKey(69), shape=(1, 1, 1)).astype(
            'bfloat16')
        self.assertTrue(np.isfinite(losses.bce_trans_loss(transitions, expected_transitions,
                                                          losses.make_prefix_nt_mask(1, 1, 1))))

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
        black_embeds = jnp.reshape(black_embeds, (1, 2, 6, 3, 3))
        metrics_data = losses.update_k_step_losses(main.FLAGS, go_model, params, i=0, data={
            'nt_original_embeds': black_embeds, 'nt_embeds': black_embeds,
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
        black_embeds = jnp.reshape(black_embeds, (1, 3, 6, 3, 3))
        metrics_data = losses.update_k_step_losses(main.FLAGS, go_model, params, i=0, data={
            'nt_original_embeds': black_embeds, 'nt_embeds': black_embeds,
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
        black_embeds = jnp.reshape(black_embeds, (1, 2, 6, 3, 3))
        metrics_data = losses.update_k_step_losses(main.FLAGS, go_model, params, i=1, data={
            'nt_original_embeds': black_embeds, 'nt_embeds': black_embeds,
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
        black_embeds = jnp.reshape(black_embeds, (2, 2, 6, 3, 3))
        metrics_data = losses.update_k_step_losses(main.FLAGS, go_model, params, i=0, data={
            'nt_original_embeds': black_embeds, 'nt_embeds': black_embeds,
            'nt_actions': jnp.array([[4, 4], [5, 5]]),
            'nt_game_winners': jnp.array([[1, -1], [1, -1]]), 'cum_val_loss': 0,
            'cum_policy_loss': 0, 'cum_trans_loss': 0,
        })
        self.assertIn('cum_trans_loss', metrics_data)
        self.assertEqual(metrics_data['cum_trans_loss'], 0)

    def test_compute_1_step_losses_black_perspective(self):
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --embed_model=black_perspective --value_model=linear '
                   '--policy_model=linear --transition_model=black_perspective'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        nt_states = gojax.decode_states("""
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
        trajectories = {
            'nt_states': jnp.reshape(nt_states, (2, 2, 6, 3, 3)),
            'nt_actions': jnp.array([[4, -1], [9, -1]], dtype='uint16')
        }
        metrics_data = losses.compute_k_step_losses(main.FLAGS, go_model, params, trajectories)
        self.assertIn('cum_trans_loss', metrics_data)
        self.assertEqual(metrics_data['cum_trans_loss'], 0)

    def test_aggregate_k_step_losses_with_trans_loss(self):
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --embed_model=cnn_lite --value_model=linear '
                   '--policy_model=linear --transition_model=cnn_lite '
                   '--add_trans_loss=false'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        nt_states = jnp.reshape(gojax.new_states(board_size=3, batch_size=4), (2, 2, 6, 3, 3))
        trajectories = {
            'nt_states': jnp.ones_like(nt_states), 'nt_actions': jnp.ones((2, 2), dtype='uint16')
        }
        loss_without_trans_loss, _ = losses.aggregate_k_step_losses(main.FLAGS, go_model, params,
                                                                    trajectories)
        main.FLAGS.add_trans_loss = True
        loss_with_trans_loss, _ = losses.aggregate_k_step_losses(main.FLAGS, go_model, params,
                                                                 trajectories)
        self.assertGreater(loss_with_trans_loss, loss_without_trans_loss)

    if __name__ == '__main__':
        unittest.main()
