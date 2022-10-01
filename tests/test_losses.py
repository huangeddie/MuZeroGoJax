"""Tests losses.py."""
# pylint: disable=missing-function-docstring,no-self-use,no-value-for-parameter,too-many-public-methods,duplicate-code

import functools
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
from muzero_gojax import nt_utils


def test_get_flat_trans_logits_with_fixed_input_no_embed_gradient_through_params():
    main.FLAGS.unparse_flags()
    main.FLAGS('foo --board_size=3 --embed_model=linear_conv --value_model=linear '
               '--policy_model=linear --transition_model=linear_conv'.split())
    go_model = models.make_model(main.FLAGS)
    states = jnp.ones_like(gojax.new_states(board_size=3, batch_size=1))
    params = go_model.init(jax.random.PRNGKey(42), states)
    embed_model = go_model.apply[models.EMBED_INDEX]
    transition_model = go_model.apply[models.TRANSITION_INDEX]
    nt_embed = jnp.reshape(embed_model(params, None, states), (1, 1, main.FLAGS.embed_dim, 3, 3))
    grads = jax.grad(lambda transition_model_, params_, nt_embed_: jnp.sum(
        losses.get_flat_trans_logits(transition_model_, params_, nt_embed_)), argnums=1)(
        transition_model, params, nt_embed)

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
    embed_model = go_model.apply[models.EMBED_INDEX]
    transition_model = go_model.apply[models.TRANSITION_INDEX]
    nt_embed = jnp.reshape(embed_model(params, None, states), (1, 1, main.FLAGS.embed_dim, 3, 3))
    grads = jax.grad(lambda transition_model_, params_, nt_embed_: jnp.sum(
        losses.get_flat_trans_logits(transition_model_, params_, nt_embed_)), argnums=2)(
        transition_model, params, nt_embed)

    np.testing.assert_array_equal(grads, jnp.zeros_like(grads))


class LossesTestCase(chex.TestCase):
    """Test losses.py"""

    @parameterized.named_parameters(('low_loss', [[[1, 0]]], [[[-1, 0]]], 0.582203),
                                    ('mid_loss', [[[0, 0]]], [[[-1, 0]]], 0.693147),
                                    ('high_loss', [[[0, 1]]], [[[-1, 0]]], 1.04432))
    def test_compute_policy_loss_from_transition_values_output(self, policy_output, value_output,
                                                               expected_loss):
        """Tests the compute_policy_loss."""
        nt_mask = nt_utils.make_suffix_nt_mask(batch_size=1, total_steps=1, suffix_len=1)
        np.testing.assert_allclose(
            losses.compute_policy_loss(jnp.array(policy_output), jnp.array(value_output), nt_mask,
                                       temp=1), expected_loss, rtol=1e-6)

    def test_update_cum_val_loss_is_type_bfloat16(self):
        """Tests output of update_cum_val_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --value_model=linear_conv'.split())
        states = jnp.ones((1, 6, 3, 3), dtype=bool)
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=states)
        params = jax.tree_util.tree_map(lambda p: jnp.ones_like(p), params)
        nt_mask = nt_utils.make_suffix_nt_mask(batch_size=1, total_steps=1, suffix_len=1)
        data = {
            'nt_game_winners': jnp.ones((1, 1), dtype='bfloat16'),
            'nt_curr_embeds': jnp.expand_dims(states, 0), **losses.initialize_metrics()
        }
        self.assertEqual(
            losses.update_cum_value_loss(go_model, params, data, nt_mask)['cum_val_loss'].dtype,
            jax.dtypes.bfloat16)

    def test_update_cum_value_low_loss(self):
        """Tests output of compute_value_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=5 --embed_model=identity --value_model=linear_conv '
                   '--hypo_steps=1'.split())
        go_model = models.make_model(main.FLAGS)
        states = jnp.ones((1, 6, 5, 5), dtype=bool)
        params = go_model.init(jax.random.PRNGKey(42), states=states)
        params = jax.tree_util.tree_map(lambda p: jnp.ones_like(p), params)
        data = {
            'nt_curr_embeds': jnp.expand_dims(states, 1), 'nt_game_winners': jnp.ones((1, 1)),
            **losses.initialize_metrics()
        }
        nt_suffix_mask = nt_utils.make_suffix_nt_mask(batch_size=1, total_steps=1, suffix_len=1)
        self.assertAlmostEqual(
            losses.update_cum_value_loss(go_model, params, data, nt_suffix_mask)['cum_val_loss'],
            9.48677e-19)
        self.assertAlmostEqual(
            losses.update_cum_value_loss(go_model, params, data, nt_suffix_mask)['cum_val_acc'], 2)

    def test_update_cum_value_high_loss(self):
        """Tests output of compute_value_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=5 --embed_model=identity --value_model=linear_conv'.split())
        go_model = models.make_model(main.FLAGS)
        states = jnp.ones((1, 6, 5, 5), dtype=bool)
        params = go_model.init(jax.random.PRNGKey(42), states=states)
        params = jax.tree_util.tree_map(lambda p: jnp.ones_like(p), params)
        data = {
            'nt_curr_embeds': jnp.expand_dims(states, 1), 'nt_game_winners': -jnp.ones((1, 1)),
            **losses.initialize_metrics()
        }
        nt_suffix_mask = nt_utils.make_suffix_nt_mask(batch_size=1, total_steps=1, suffix_len=1)
        self.assertAlmostEqual(
            losses.update_cum_value_loss(go_model, params, data, nt_suffix_mask)['cum_val_loss'],
            41.5)
        self.assertAlmostEqual(
            losses.update_cum_value_loss(go_model, params, data, nt_suffix_mask)['cum_val_acc'], 0)

    def test_update_cum_value_loss_nan(self):
        """Tests output of compute_value_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=5 --embed_model=identity --value_model=linear_conv'.split())
        go_model = models.make_model(main.FLAGS)
        states = jnp.ones((1, 6, 5, 5), dtype=bool)
        params = go_model.init(jax.random.PRNGKey(42), states=states)
        params = jax.tree_util.tree_map(lambda p: jnp.ones_like(p), params)
        data = {
            'nt_curr_embeds': jnp.expand_dims(states, 1), 'nt_game_winners': jnp.ones((1, 1)),
            **losses.initialize_metrics()
        }
        nt_suffix_mask = nt_utils.make_suffix_nt_mask(batch_size=1, total_steps=1, suffix_len=0)
        self.assertTrue(jnp.isnan(
            losses.update_cum_value_loss(go_model, params, data, nt_suffix_mask)['cum_val_loss']))
        self.assertAlmostEqual(
            losses.update_cum_value_loss(go_model, params, data, nt_suffix_mask)['cum_val_acc'], 0)

    def test_update_decode_loss_low_loss(self):
        """Tests output of decode_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=5 --embed_model=identity --decode_model=linear_conv --hdim=8 '
                   '--nlayers=1 --hypo_steps=1'.split())
        go_model = models.make_model(main.FLAGS)
        states = jnp.ones((1, 6, 3, 3), dtype=bool)
        params = go_model.init(jax.random.PRNGKey(42), states=states)
        params = jax.tree_util.tree_map(lambda p: jnp.ones_like(p), params)
        data = {
            'nt_states': jnp.expand_dims(states, 1), 'nt_curr_embeds': jnp.expand_dims(states, 1),
            **losses.initialize_metrics()
        }
        nt_suffix_mask = nt_utils.make_suffix_nt_mask(batch_size=1, total_steps=1, suffix_len=1)
        self.assertAlmostEqual(
            losses.update_cum_decode_loss(go_model, params, data, nt_suffix_mask)[
                'cum_decode_loss'], 3.32875e-10)
        self.assertAlmostEqual(
            losses.update_cum_decode_loss(go_model, params, data, nt_suffix_mask)['cum_decode_acc'],
            2)

    def test_update_decode_loss_high_loss(self):
        """Tests output of decode_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=5 --embed_model=identity --decode_model=linear_conv --hdim=8 '
                   '--nlayers=1 --hypo_steps=1'.split())
        go_model = models.make_model(main.FLAGS)
        states = jnp.zeros((1, 6, 3, 3), dtype=bool)
        params = go_model.init(jax.random.PRNGKey(42), states=states)
        params = jax.tree_util.tree_map(lambda p: jnp.ones_like(p), params)
        data = {
            'nt_states': jnp.expand_dims(states, 1), 'nt_curr_embeds': jnp.expand_dims(states, 1),
            **losses.initialize_metrics()
        }
        nt_suffix_mask = nt_utils.make_suffix_nt_mask(batch_size=1, total_steps=1, suffix_len=1)
        self.assertAlmostEqual(
            losses.update_cum_decode_loss(go_model, params, data, nt_suffix_mask)[
                'cum_decode_loss'], 71)
        self.assertAlmostEqual(
            losses.update_cum_decode_loss(go_model, params, data, nt_suffix_mask)['cum_decode_acc'],
            0)

    def test_initialize_metrics(self):
        initial_metrics = losses.initialize_metrics()
        self.assertIsInstance(initial_metrics, dict)
        self.assertLen(initial_metrics, 7)

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
        nt_black_embeds = jnp.reshape(black_embeds, (1, 2, 6, 3, 3))
        metrics_data = losses.update_k_step_losses(main.FLAGS, go_model, params, i=0, data={
            'nt_states': nt_black_embeds, 'nt_original_embeds': nt_black_embeds,
            'nt_curr_embeds': nt_black_embeds, 'flattened_actions': jnp.array([4, 4]),
            'nt_game_winners': jnp.array([[1, -1]]), **losses.initialize_metrics()
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
        nt_black_embeds = jnp.reshape(black_embeds, (1, 3, 6, 3, 3))
        metrics_data = losses.update_k_step_losses(main.FLAGS, go_model, params, i=0, data={
            'nt_states': nt_black_embeds, 'nt_original_embeds': nt_black_embeds,
            'nt_curr_embeds': nt_black_embeds, 'flattened_actions': jnp.array([8, 6, 6]),
            'nt_game_winners': jnp.array([[0, 0, 0]]), **losses.initialize_metrics()
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
        nt_black_embeds = jnp.reshape(black_embeds, (1, 2, 6, 3, 3))
        metrics_data = losses.update_k_step_losses(main.FLAGS, go_model, params, i=1, data={
            'nt_states': nt_black_embeds, 'nt_original_embeds': nt_black_embeds,
            'nt_curr_embeds': nt_black_embeds, 'flattened_actions': jnp.array([4, 4]),
            'nt_game_winners': jnp.array([[1, -1]]), **losses.initialize_metrics()
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
        nt_black_embeds = jnp.reshape(black_embeds, (2, 2, 6, 3, 3))
        metrics_data = losses.update_k_step_losses(main.FLAGS, go_model, params, i=0, data={
            'nt_states': nt_black_embeds, 'nt_original_embeds': nt_black_embeds,
            'nt_curr_embeds': nt_black_embeds, 'flattened_actions': jnp.array([4, 4, 5, 5]),
            'nt_game_winners': jnp.array([[1, -1], [1, -1]]), **losses.initialize_metrics()
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
        main.FLAGS('foo --board_size=3 --embed_model=cnn_lite --value_model=linear_conv '
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

    def test_aggregate_k_step_losses_accuracy_keys(self):
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --embed_model=linear_conv --value_model=linear_conv '
                   '--policy_model=linear_conv --transition_model=linear_conv '
                   '--monitor_trans_acc=true'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        nt_states = jnp.reshape(gojax.new_states(board_size=3, batch_size=1), (1, 1, 6, 3, 3))
        trajectories = {
            'nt_states': nt_states, 'nt_actions': jnp.ones((1, 1), dtype='uint16')
        }
        _, metric_data = losses.aggregate_k_step_losses(main.FLAGS, go_model, params, trajectories)
        self.assertIn('trans_acc', metric_data)
        self.assertNotIn('cum_trans_acc', metric_data)
        self.assertIn('val_acc', metric_data)
        self.assertNotIn('cum_val_acc', metric_data)
        self.assertIn('decode_acc', metric_data)
        self.assertNotIn('cum_decode_acc', metric_data)

    def test_aggregate_k_step_losses_with_monitor_trans_acc(self):
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --embed_model=identity --value_model=linear_conv '
                   '--policy_model=linear --transition_model=cnn_lite --embed_dim=6 '
                   '--monitor_trans_acc=true'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        nt_states = jnp.reshape(gojax.new_states(board_size=3, batch_size=4), (2, 2, 6, 3, 3))
        trajectories = {
            'nt_states': jnp.ones_like(nt_states), 'nt_actions': jnp.ones((2, 2), dtype='uint16')
        }
        _, metric_data = losses.aggregate_k_step_losses(main.FLAGS, go_model, params, trajectories)
        self.assertIn('trans_acc', metric_data)

    def test_aggregate_k_step_losses_no_monitor_trans_acc(self):
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --embed_model=identity --value_model=linear_conv '
                   '--policy_model=linear --transition_model=cnn_lite --embed_dim=6 '
                   '--monitor_trans_acc=false'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        nt_states = jnp.reshape(gojax.new_states(board_size=3, batch_size=4), (2, 2, 6, 3, 3))
        trajectories = {
            'nt_states': jnp.ones_like(nt_states), 'nt_actions': jnp.ones((2, 2), dtype='uint16')
        }
        _, metric_data = losses.aggregate_k_step_losses(main.FLAGS, go_model, params, trajectories)
        self.assertNotIn('trans_acc', metric_data)

    def test_compute_loss_gradients_yields_negative_value_gradients(self):
        """
        Given a model with positive parameters and a single won state, check that the value
        parameter gradients are negative.
        """
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --hdim=2 --embed_model=linear_conv --value_model=linear '
                   '--policy_model=linear --transition_model=linear_conv --hypo_steps=1'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        params = jax.tree_util.tree_map(lambda x: jnp.full_like(x, 1e-3), params)
        nt_states = gojax.decode_states("""
                                            _ _ _
                                            _ B _
                                            _ _ _
                                            """)
        trajectories = {
            'nt_states': jnp.reshape(nt_states, (1, 1, 6, 3, 3)),
            'nt_actions': jnp.full((1, 1), fill_value=-1, dtype='uint16')
        }
        grads, _ = losses.compute_loss_gradients_and_metrics(main.FLAGS, go_model, params,
                                                             trajectories)
        self.assertIn('linear3_d_value', grads)
        self.assertIn('value_w', grads['linear3_d_value'])
        self.assertIn('value_b', grads['linear3_d_value'])
        np.testing.assert_array_less(grads['linear3_d_value']['value_w'],
                                     jnp.zeros_like(grads['linear3_d_value']['value_w']))
        np.testing.assert_array_less(grads['linear3_d_value']['value_b'],
                                     jnp.zeros_like(grads['linear3_d_value']['value_b']))

    def test_compute_loss_gradients_yields_positive_value_gradients(self):
        """
        Given a model with positive parameters and a single loss state, check that the value
        parameter gradients are positive.
        """
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --hdim=2 --embed_model=linear_conv --value_model=linear '
                   '--policy_model=linear --transition_model=linear_conv --hypo_steps=1'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        params = jax.tree_util.tree_map(lambda x: jnp.full_like(x, 1e-3), params)
        nt_states = gojax.decode_states("""
                                            _ _ _
                                            _ W _
                                            _ _ _
                                            """)
        trajectories = {
            'nt_states': jnp.reshape(nt_states, (1, 1, 6, 3, 3)),
            'nt_actions': jnp.full((1, 1), fill_value=-1, dtype='uint16')
        }
        grads, _ = losses.compute_loss_gradients_and_metrics(main.FLAGS, go_model, params,
                                                             trajectories)
        self.assertIn('linear3_d_value', grads)
        self.assertIn('value_w', grads['linear3_d_value'])
        self.assertIn('value_b', grads['linear3_d_value'])
        np.testing.assert_array_less(-grads['linear3_d_value']['value_w'],
                                     jnp.zeros_like(grads['linear3_d_value']['value_w']))
        np.testing.assert_array_less(-grads['linear3_d_value']['value_b'],
                                     jnp.zeros_like(grads['linear3_d_value']['value_b']))

    def test_compute_loss_gradients_with_one_step_and_no_trans_loss_has_non_transition_grads(self):
        """Tests all parameters except for transitions have grads with compute_0_step_total_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --hdim=2 --embed_model=linear_conv --value_model=linear '
                   '--policy_model=linear --transition_model=linear_conv --hypo_steps=1 '
                   '--add_trans_loss=false'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        trajectories = {
            'nt_states': jnp.ones((1, 1, 6, 3, 3), dtype=bool),
            'nt_actions': jnp.ones((1, 1), dtype='uint16')
        }
        grads, _ = losses.compute_loss_gradients_and_metrics(main.FLAGS, go_model, params,
                                                             trajectories)

        # Check everything except transition grads is non-zero.
        del grads['linear_conv_transition/~/conv2_d']['b']
        del grads['linear_conv_transition/~/conv2_d']['w']
        self.assertTrue(functools.reduce(lambda a, b: a and b,
                                         map(lambda grad: grad.astype(bool).all(),
                                             jax.tree_util.tree_flatten(grads)[0])))

    def test_compute_loss_gradients_with_one_step_and_no_trans_loss_has_zero_transition_grads(self):
        """Tests all parameters except for transitions have grads with compute_0_step_total_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --hdim=2 --embed_model=linear_conv --value_model=linear '
                   '--policy_model=linear --transition_model=linear_conv --hypo_steps=1 '
                   '--add_trans_loss=false'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        trajectories = {
            'nt_states': jnp.ones((1, 1, 6, 3, 3), dtype=bool),
            'nt_actions': jnp.ones((1, 1), dtype='uint16')
        }
        grads, _ = losses.compute_loss_gradients_and_metrics(main.FLAGS, go_model, params,
                                                             trajectories)

        # Check all transition weights are 0.
        self.assertFalse(grads['linear_conv_transition/~/conv2_d']['b'].astype(bool).any())
        self.assertFalse(grads['linear_conv_transition/~/conv2_d']['w'].astype(bool).any())

    def test_compute_loss_gradients_with_one_step_and_trans_loss_has_nonzero_transition_grads(self):
        """Tests all parameters except for transitions have grads with compute_0_step_total_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --hdim=2 --embed_model=linear_conv --value_model=linear '
                   '--policy_model=linear --transition_model=linear_conv --hypo_steps=1 '
                   '--add_trans_loss=true'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        trajectories = {
            'nt_states': jnp.ones((1, 1, 6, 3, 3), dtype=bool),
            'nt_actions': jnp.ones((1, 1), dtype='uint16')
        }
        grads, _ = losses.compute_loss_gradients_and_metrics(main.FLAGS, go_model, params,
                                                             trajectories)

        # Check all transition weights are non-zero.
        self.assertTrue(grads['linear_conv_transition/~/conv2_d']['b'].astype(bool).any())
        self.assertTrue(grads['linear_conv_transition/~/conv2_d']['w'].astype(bool).any())

    def test_compute_loss_gradients_with_two_steps_and_trans_loss_has_nonzero_grads(self):
        """Tests all parameters except for transitions have grads with compute_0_step_total_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --hdim=2 --embed_model=linear_conv --value_model=linear '
                   '--policy_model=linear --transition_model=linear_conv --hypo_steps=2 '
                   '--add_trans_loss=true'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        trajectories = {
            'nt_states': jnp.ones((1, 2, 6, 3, 3), dtype=bool),
            'nt_actions': jnp.ones((1, 2), dtype='uint16')
        }
        grads, _ = losses.compute_loss_gradients_and_metrics(main.FLAGS, go_model, params,
                                                             trajectories)

        # Check some transition weights are non-zero.
        self.assertTrue(grads['linear_conv_transition/~/conv2_d']['b'].astype(bool).any())
        self.assertTrue(grads['linear_conv_transition/~/conv2_d']['w'].astype(bool).any())
        # Check everything else is non-zero.
        del grads['linear_conv_transition/~/conv2_d']['b']
        del grads['linear_conv_transition/~/conv2_d']['w']
        self.assertTrue(functools.reduce(lambda a, b: a and b,
                                         map(lambda grad: grad.astype(bool).all(),
                                             jax.tree_util.tree_flatten(grads)[0])))

    def test_compute_loss_gradients_with_two_steps_and_no_trans_loss_has_no_trans_grads(self):
        """Tests all parameters except for transitions have grads with compute_0_step_total_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --hdim=2 --embed_model=linear_conv --value_model=linear '
                   '--policy_model=linear --transition_model=linear_conv --hypo_steps=2 '
                   '--add_trans_loss=false'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        trajectories = {
            'nt_states': jnp.ones((1, 2, 6, 3, 3), dtype=bool),
            'nt_actions': jnp.ones((1, 2), dtype='uint16')
        }
        grads, _ = losses.compute_loss_gradients_and_metrics(main.FLAGS, go_model, params,
                                                             trajectories)

        # Check some transition weights are non-zero.
        self.assertFalse(grads['linear_conv_transition/~/conv2_d']['b'].astype(bool).any())
        self.assertFalse(grads['linear_conv_transition/~/conv2_d']['w'].astype(bool).any())

    if __name__ == '__main__':
        unittest.main()
