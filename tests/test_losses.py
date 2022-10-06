"""Tests losses.py."""
# pylint: disable=missing-function-docstring,no-value-for-parameter,too-many-public-methods,duplicate-code

import functools
import unittest
from typing import Union

import chex
import gojax
import jax.random
import numpy as np
from absl import flags
from jax import numpy as jnp

from muzero_gojax import game
from muzero_gojax import losses
from muzero_gojax import main
from muzero_gojax import models
from muzero_gojax import nt_utils


def mock_initial_data(absl_flags: flags.FlagValues, embed_fill_value: Union[int, float] = 0,
                      embed_dtype: str = 'bool', transition_logit_fill_value: Union[int, float] = 0,
                      transition_dtype: str = 'bool') -> losses.LossData:
    """Mocks the initial data in compute_k_step_losses."""
    # pylint: disable=too-many-arguments
    states = gojax.new_states(absl_flags.board_size, absl_flags.batch_size)
    nt_states = jnp.repeat(jnp.expand_dims(states, 0), absl_flags.trajectory_length, axis=1)
    embeddings = jnp.full_like(nt_states, embed_fill_value, dtype=embed_dtype)
    transition_logits = jnp.repeat(jnp.expand_dims(
        jnp.full_like(nt_states, transition_logit_fill_value, dtype=transition_dtype), axis=2),
        gojax.get_action_size(states), axis=2)
    nt_actions = jnp.zeros(absl_flags.batch_size * absl_flags.trajectory_length, dtype='uint8')
    trajectories = game.Trajectories(nt_states, nt_actions)
    nt_game_winners = jnp.zeros((absl_flags.batch_size, absl_flags.trajectory_length))
    return losses.LossData(trajectories, embeddings, embeddings, transition_logits, nt_game_winners)


class LossesTestCase(chex.TestCase):
    """Test losses.py"""

    def assertPytreeAllZero(self, pytree):
        # pylint: disable=invalid-name
        """Asserts all leaves in the pytree are zero."""
        if not functools.reduce(lambda a, b: a and b, map(lambda grad: (~grad.astype(bool)).all(),
                                                          jax.tree_util.tree_flatten(pytree)[0])):
            raise AssertionError(f"PyTree has non-zero elements: {pytree}")

    def assertPytreeAnyNonZero(self, pytree):
        # pylint: disable=invalid-name
        """Asserts all leaves in the pytree are zero."""
        if not functools.reduce(lambda a, b: a or b, map(lambda grad: grad.astype(bool).any(),
                                                         jax.tree_util.tree_flatten(pytree)[0])):
            raise AssertionError(f"PyTree no non-zero elements: {pytree}")

    def assertPytreeAllNonZero(self, pytree):
        # pylint: disable=invalid-name
        """Asserts all leaves in the pytree are non-zero."""
        if not functools.reduce(lambda a, b: a and b, map(lambda grad: grad.astype(bool).all(),
                                                          jax.tree_util.tree_flatten(pytree)[0])):
            raise AssertionError(f"PyTree has zero elements: {pytree}")

    def test_assert_pytree_all_zero(self):
        self.assertPytreeAllZero({'a': jnp.zeros(()), 'b': {'c': jnp.zeros(2)}})
        with self.assertRaises(AssertionError):
            self.assertPytreeAllZero({'a': jnp.zeros(()), 'b': {'c': jnp.array([0, 1])}})

    def test_assert_pytree_any_non_zero(self):
        self.assertPytreeAnyNonZero({'a': jnp.zeros(()), 'b': {'c': jnp.array([0, 1])}})
        with self.assertRaises(AssertionError):
            self.assertPytreeAnyNonZero({'a': jnp.zeros(()), 'b': {'c': jnp.zeros(2)}})

    def test_assert_pytree_all_non_zero(self):
        self.assertPytreeAllNonZero({'a': jnp.ones(()), 'b': {'c': jnp.ones(2)}})
        with self.assertRaises(AssertionError):
            self.assertPytreeAllNonZero({'a': jnp.ones(()), 'b': {'c': jnp.array([0, 1])}})

    def test_update_cum_policy_loss_low_loss(self):
        """Tests the update_cum_policy_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --batch_size=1 --board_size=3 --trajectory_length=1 --temperature=1 '
                   '--embed_model=linear_conv --value_model=linear_conv --policy_model=linear '
                   '--transition_model=linear_conv'.split())
        go_model = models.make_model(main.FLAGS)
        states = gojax.new_states(main.FLAGS.board_size)
        params = jax.tree_util.tree_map(lambda x: jnp.ones_like(x),
                                        go_model.init(jax.random.PRNGKey(42), states))
        # Make the policy output high value for first action.
        params['linear3_d_policy']['action_b'] = params['linear3_d_policy']['action_b'].at[
            0, 0].set(10)
        nt_mask = nt_utils.make_suffix_nt_mask(batch_size=main.FLAGS.batch_size,
                                               total_steps=main.FLAGS.trajectory_length,
                                               suffix_len=main.FLAGS.trajectory_length)
        data = mock_initial_data(main.FLAGS, embed_dtype='bfloat16', transition_dtype='bfloat16')
        # Make value model output low value (high target) for first action.
        data = data._replace(nt_transition_logits=data.nt_transition_logits.at[0, 0, 0].set(-1))
        np.testing.assert_allclose(losses.update_cum_policy_loss(main.FLAGS, go_model, params, data,
                                                                 nt_mask).cum_policy_loss, 0.00111,
                                   atol=1e-6)

    def test_update_cum_policy_loss_high_loss(self):
        """Tests the update_cum_policy_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --batch_size=1 --board_size=3 --trajectory_length=1 --temperature=1 '
                   '--embed_model=linear_conv --value_model=linear_conv --policy_model=linear '
                   '--transition_model=linear_conv'.split())
        go_model = models.make_model(main.FLAGS)
        states = gojax.new_states(main.FLAGS.board_size)
        params = jax.tree_util.tree_map(lambda x: jnp.ones_like(x),
                                        go_model.init(jax.random.PRNGKey(42), states))
        # Make the policy output high value for first action.
        params['linear3_d_policy']['action_b'] = params['linear3_d_policy']['action_b'].at[
            0, 0].set(10)
        nt_mask = nt_utils.make_suffix_nt_mask(batch_size=main.FLAGS.batch_size,
                                               total_steps=main.FLAGS.trajectory_length,
                                               suffix_len=main.FLAGS.trajectory_length)
        data = mock_initial_data(main.FLAGS, embed_dtype='bfloat16', transition_dtype='bfloat16')
        # Make value model output high value (low target) for first action.
        data = data._replace(nt_transition_logits=data.nt_transition_logits.at[0, 0, 0].set(1))
        np.testing.assert_allclose(losses.update_cum_policy_loss(main.FLAGS, go_model, params, data,
                                                                 nt_mask).cum_policy_loss, 9.018691,
                                   rtol=1e-5)

    def test_get_next_hypo_embed_logits(self):
        """Tests get_next_hypo_embed_logits gets the right embeddings from the transitions."""
        nt_transition_logits = jnp.zeros((1, 2, 4, 6, 3, 3))
        nt_transition_logits = nt_transition_logits.at[0, 1, 1].set(1)
        next_hypo_embed_logits = losses.get_next_hypo_embed_logits(
            losses.LossData(trajectories=game.Trajectories(nt_actions=jnp.arange(2)),
                            nt_transition_logits=nt_transition_logits))
        np.testing.assert_array_equal(next_hypo_embed_logits[0, 0],
                                      jnp.ones_like(next_hypo_embed_logits[0, 0]))
        np.testing.assert_array_equal(next_hypo_embed_logits[0, 1],
                                      jnp.zeros_like(next_hypo_embed_logits[0, 1]))

    def test_update_curr_embeds_updates_nt_curr_embeds(self):
        """Tests output of update_curr_embeds."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --trajectory_length=1'.split())
        data = mock_initial_data(main.FLAGS, transition_logit_fill_value=1)
        data = losses.update_curr_embeds(main.FLAGS, data)
        next_embeds = data.nt_curr_embeds
        np.testing.assert_array_equal(next_embeds, jnp.ones_like(next_embeds))

    def test_update_curr_embeds_cuts_gradient(self):
        """Tests output of update_curr_embeds."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --trajectory_length=1'.split())
        data = mock_initial_data(main.FLAGS, transition_logit_fill_value=1,
                                 transition_dtype='bfloat16')

        def grad_fn(data_):
            """Sums the nt_curr_embeds."""
            return jnp.sum(losses.update_curr_embeds(main.FLAGS, data_).nt_curr_embeds)

        grad_data = jax.grad(grad_fn, allow_int=True)(data)
        np.testing.assert_array_equal(grad_data.nt_transition_logits,
                                      jnp.zeros_like(grad_data.nt_transition_logits))

    def test_update_cum_val_loss_is_type_bfloat16(self):
        """Tests output of update_cum_val_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --value_model=linear_conv'.split())
        states = jnp.ones((1, 6, 3, 3), dtype=bool)
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=states)
        params = jax.tree_util.tree_map(lambda p: jnp.ones_like(p), params)
        nt_mask = nt_utils.make_suffix_nt_mask(batch_size=1, total_steps=1, suffix_len=1)
        data = losses.LossData(nt_curr_embeds=jnp.expand_dims(states, 0),
                               nt_game_winners=jnp.ones((1, 1), dtype='bfloat16'))
        self.assertEqual(
            losses.update_cum_value_loss(go_model, params, data, nt_mask).cum_val_loss.dtype,
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
        data = losses.LossData(nt_curr_embeds=jnp.expand_dims(states, 1),
                               nt_game_winners=jnp.ones((1, 1)))
        nt_suffix_mask = nt_utils.make_suffix_nt_mask(batch_size=1, total_steps=1, suffix_len=1)
        self.assertAlmostEqual(
            losses.update_cum_value_loss(go_model, params, data, nt_suffix_mask).cum_val_loss,
            9.48677e-19)
        self.assertAlmostEqual(
            losses.update_cum_value_loss(go_model, params, data, nt_suffix_mask).cum_val_acc, 1)

    def test_update_cum_value_high_loss(self):
        """Tests output of compute_value_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=5 --embed_model=identity --value_model=linear_conv'.split())
        go_model = models.make_model(main.FLAGS)
        states = jnp.ones((1, 6, 5, 5), dtype=bool)
        params = go_model.init(jax.random.PRNGKey(42), states=states)
        params = jax.tree_util.tree_map(lambda p: jnp.ones_like(p), params)
        data = losses.LossData(nt_curr_embeds=jnp.expand_dims(states, 1),
                               nt_game_winners=-jnp.ones((1, 1)))
        nt_suffix_mask = nt_utils.make_suffix_nt_mask(batch_size=1, total_steps=1, suffix_len=1)
        self.assertAlmostEqual(
            losses.update_cum_value_loss(go_model, params, data, nt_suffix_mask).cum_val_loss, 41.5)
        self.assertAlmostEqual(
            losses.update_cum_value_loss(go_model, params, data, nt_suffix_mask).cum_val_acc, 0)

    def test_update_cum_value_loss_nan(self):
        """Tests output of compute_value_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=5 --embed_model=identity --value_model=linear_conv'.split())
        go_model = models.make_model(main.FLAGS)
        states = jnp.ones((1, 6, 5, 5), dtype=bool)
        params = go_model.init(jax.random.PRNGKey(42), states=states)
        params = jax.tree_util.tree_map(lambda p: jnp.ones_like(p), params)
        data = losses.LossData(nt_curr_embeds=jnp.expand_dims(states, 1),
                               nt_game_winners=jnp.ones((1, 1)))
        nt_suffix_mask = nt_utils.make_suffix_nt_mask(batch_size=1, total_steps=1, suffix_len=0)
        self.assertTrue(jnp.isnan(
            losses.update_cum_value_loss(go_model, params, data, nt_suffix_mask).cum_val_loss))
        self.assertAlmostEqual(
            losses.update_cum_value_loss(go_model, params, data, nt_suffix_mask).cum_val_acc, 0)

    def test_update_decode_loss_low_loss(self):
        """Tests output of decode_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=5 --embed_model=identity --decode_model=linear_conv --hdim=8 '
                   '--nlayers=1 --hypo_steps=1'.split())
        go_model = models.make_model(main.FLAGS)
        states = jnp.ones((1, 6, 3, 3), dtype=bool)
        params = go_model.init(jax.random.PRNGKey(42), states=states)
        params = jax.tree_util.tree_map(lambda p: jnp.ones_like(p), params)
        data = losses.LossData(trajectories=game.Trajectories(nt_states=jnp.expand_dims(states, 1)),
                               nt_curr_embeds=jnp.expand_dims(states, 1))
        nt_suffix_mask = nt_utils.make_suffix_nt_mask(batch_size=1, total_steps=1, suffix_len=1)
        self.assertAlmostEqual(
            losses.update_cum_decode_loss(go_model, params, data, nt_suffix_mask).cum_decode_loss,
            3.32875e-10)
        self.assertAlmostEqual(
            losses.update_cum_decode_loss(go_model, params, data, nt_suffix_mask).cum_decode_acc, 1)

    def test_update_decode_loss_high_loss(self):
        """Tests output of decode_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=5 --embed_model=identity --decode_model=linear_conv --hdim=8 '
                   '--nlayers=1 --hypo_steps=1'.split())
        go_model = models.make_model(main.FLAGS)
        states = jnp.zeros((1, 6, 3, 3), dtype=bool)
        params = go_model.init(jax.random.PRNGKey(42), states=states)
        params = jax.tree_util.tree_map(lambda p: jnp.ones_like(p), params)
        data = losses.LossData(trajectories=game.Trajectories(nt_states=jnp.expand_dims(states, 1)),
                               nt_curr_embeds=jnp.expand_dims(states, 1))
        nt_suffix_mask = nt_utils.make_suffix_nt_mask(batch_size=1, total_steps=1, suffix_len=1)
        self.assertAlmostEqual(
            losses.update_cum_decode_loss(go_model, params, data, nt_suffix_mask).cum_decode_loss,
            71)
        self.assertAlmostEqual(
            losses.update_cum_decode_loss(go_model, params, data, nt_suffix_mask).cum_decode_acc, 0)

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
        data = losses.LossData(
            trajectories=game.Trajectories(nt_states=nt_black_embeds, nt_actions=jnp.array([4, 4])),
            nt_original_embeds=nt_black_embeds, nt_curr_embeds=nt_black_embeds,
            nt_game_winners=jnp.array([[1, -1]]))
        metrics_data = losses.update_k_step_losses(main.FLAGS, go_model, params, i=0, data=data)
        self.assertEqual(metrics_data.cum_trans_loss, 0)

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
        data = losses.LossData(trajectories=game.Trajectories(nt_states=nt_black_embeds,
                                                              nt_actions=jnp.array([8, 6, 6])),
                               nt_original_embeds=nt_black_embeds, nt_curr_embeds=nt_black_embeds,
                               nt_game_winners=jnp.array([[0, 0, 0]]))
        metrics_data = losses.update_k_step_losses(main.FLAGS, go_model, params, i=0, data=data)
        self.assertEqual(metrics_data.cum_trans_loss, 0)

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
        data = losses.LossData(trajectories=game.Trajectories(nt_black_embeds, jnp.array([4, 4])),
                               nt_original_embeds=nt_black_embeds, nt_curr_embeds=nt_black_embeds,
                               nt_game_winners=jnp.array([[1, -1]]))
        metrics_data = losses.update_k_step_losses(main.FLAGS, go_model, params, i=1, data=data)
        self.assertEqual(metrics_data.cum_trans_loss, 0)

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
        data = losses.LossData(
            trajectories=game.Trajectories(nt_black_embeds, jnp.array([4, 4, 5, 5])),
            nt_original_embeds=nt_black_embeds, nt_curr_embeds=nt_black_embeds,
            nt_game_winners=jnp.array([[1, -1], [1, -1]]))
        metrics_data = losses.update_k_step_losses(main.FLAGS, go_model, params, i=0, data=data)
        self.assertEqual(metrics_data.cum_trans_loss, 0)

    def test_compute_1_step_losses_black_perspective(self):
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --embed_model=black_perspective --embed_dim=6 ' \
                   '--value_model=linear --policy_model=linear ' \
                   '--transition_model=black_perspective'.split())
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
        trajectories = game.Trajectories(nt_states=jnp.reshape(nt_states, (2, 2, 6, 3, 3)),
                                         nt_actions=jnp.array([[4, -1], [9, -1]], dtype='uint16'))
        loss_data = losses.compute_k_step_losses(main.FLAGS, go_model, params, trajectories)
        self.assertEqual(loss_data.cum_trans_loss, 0)

    def test_aggregate_k_step_losses_with_trans_loss(self):
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --embed_model=cnn_lite --value_model=linear_conv '
                   '--policy_model=linear --transition_model=cnn_lite '
                   '--add_trans_loss=false'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        nt_states = jnp.reshape(gojax.new_states(board_size=3, batch_size=4), (2, 2, 6, 3, 3))
        trajectories = game.Trajectories(nt_states=jnp.ones_like(nt_states),
                                         nt_actions=jnp.ones((2, 2), dtype='uint16'))
        loss_without_trans_loss, _ = losses.aggregate_k_step_losses(main.FLAGS, go_model, params,
                                                                    trajectories)
        main.FLAGS.add_trans_loss = True
        loss_with_trans_loss, _ = losses.aggregate_k_step_losses(main.FLAGS, go_model, params,
                                                                 trajectories)
        self.assertGreater(loss_with_trans_loss, loss_without_trans_loss)

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
        trajectories = game.Trajectories(nt_states=jnp.reshape(nt_states, (1, 1, 6, 3, 3)),
                                         nt_actions=jnp.full((1, 1), fill_value=-1, dtype='uint16'))
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
        trajectories = game.Trajectories(nt_states=jnp.reshape(nt_states, (1, 1, 6, 3, 3)),
                                         nt_actions=jnp.full((1, 1), fill_value=-1, dtype='uint16'))
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
        trajectories = game.Trajectories(nt_states=jnp.ones((1, 1, 6, 3, 3), dtype=bool),
                                         nt_actions=jnp.ones((1, 1), dtype='uint16'))
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
        trajectories = game.Trajectories(nt_states=jnp.ones((1, 1, 6, 3, 3), dtype=bool),
                                         nt_actions=jnp.ones((1, 1), dtype='uint16'))
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
        trajectories = game.Trajectories(nt_states=jnp.ones((1, 1, 6, 3, 3), dtype=bool),
                                         nt_actions=jnp.ones((1, 1), dtype='uint16'))
        grads, _ = losses.compute_loss_gradients_and_metrics(main.FLAGS, go_model, params,
                                                             trajectories)

        # Check all transition weights are non-zero.
        self.assertTrue(grads['linear_conv_transition/~/conv2_d']['b'].astype(bool).any())
        self.assertTrue(grads['linear_conv_transition/~/conv2_d']['w'].astype(bool).any())

    def test_compute_loss_gradients_no_loss_no_gradients(self):
        """Tests all parameters except for transitions have grads with compute_0_step_total_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS(
            'foo --board_size=3 --hdim=2 --embed_model=linear_conv --value_model=linear_conv '
            '--policy_model=linear_conv --transition_model=linear_conv --hypo_steps=1 '
            '--add_value_loss=false --add_decode_loss=false --add_policy_loss=false '
            '--add_trans_loss=false'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        trajectories = game.Trajectories(nt_states=jnp.ones((1, 1, 6, 3, 3), dtype=bool),
                                         nt_actions=jnp.ones((1, 1), dtype='uint16'))
        grads, _ = losses.compute_loss_gradients_and_metrics(main.FLAGS, go_model, params,
                                                             trajectories)

        # Check all transition weights are non-zero.
        self.assertPytreeAllZero(grads)

    def test_compute_loss_gradients_transition_loss_only_affects_transition_gradients(self):
        """Tests all parameters except for transitions have grads with compute_0_step_total_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS(
            'foo --board_size=3 --hdim=2 --embed_model=linear_conv --value_model=linear_conv '
            '--policy_model=linear_conv --transition_model=linear_conv --hypo_steps=1 '
            '--add_value_loss=false --add_decode_loss=false --add_policy_loss=false '
            '--add_trans_loss=true'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        trajectories = game.Trajectories(nt_states=jnp.ones((1, 1, 6, 3, 3), dtype=bool),
                                         nt_actions=jnp.ones((1, 1), dtype='uint16'))
        grads: dict
        grads, _ = losses.compute_loss_gradients_and_metrics(main.FLAGS, go_model, params,
                                                             trajectories)

        # Check a strict subset of transition weights are non-zero.
        self.assertPytreeAnyNonZero(grads['linear_conv_transition/~/conv2_d'])
        with self.assertRaises(AssertionError):
            self.assertPytreeAllNonZero(grads['linear_conv_transition/~/conv2_d'])
        # Check the remaining gradients are zero.
        grads.pop('linear_conv_transition/~/conv2_d')
        self.assertPytreeAllZero(grads)

    def test_compute_loss_gradients_with_two_steps_and_trans_loss_has_nonzero_grads(self):
        """Tests all parameters except for transitions have grads with compute_0_step_total_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --hdim=2 --embed_model=linear_conv --value_model=linear '
                   '--policy_model=linear --transition_model=linear_conv --hypo_steps=2 '
                   '--add_trans_loss=true'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        trajectories = game.Trajectories(nt_states=jnp.ones((1, 2, 6, 3, 3), dtype=bool),
                                         nt_actions=jnp.ones((1, 2), dtype='uint16'))
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
        trajectories = game.Trajectories(nt_states=jnp.ones((1, 2, 6, 3, 3), dtype=bool),
                                         nt_actions=jnp.ones((1, 2), dtype='uint16'))
        grads, _ = losses.compute_loss_gradients_and_metrics(main.FLAGS, go_model, params,
                                                             trajectories)

        # Check some transition weights are non-zero.
        self.assertFalse(grads['linear_conv_transition/~/conv2_d']['b'].astype(bool).any())
        self.assertFalse(grads['linear_conv_transition/~/conv2_d']['w'].astype(bool).any())

    if __name__ == '__main__':
        unittest.main()
