"""Tests losses.py."""
# pylint: disable=missing-function-docstring,no-value-for-parameter,too-many-public-methods,duplicate-code

import functools
from typing import Union

import chex
import gojax
import jax.random
import numpy as np
from absl.testing import absltest
from absl.testing import flagsaver
from jax import numpy as jnp

from muzero_gojax import game
from muzero_gojax import losses
from muzero_gojax import main
from muzero_gojax import models

FLAGS = main.FLAGS


def _make_zeros_like_loss_data(board_size: int, batch_size: int, trajectory_length: int):
    states = gojax.new_states(board_size, batch_size)
    nt_states = jnp.repeat(jnp.expand_dims(states, 0), trajectory_length, axis=1)
    nt_actions = jnp.zeros(batch_size * trajectory_length, dtype='uint8')
    embeddings = jnp.zeros_like(nt_states, dtype='bfloat16')
    transition_logits = jnp.repeat(
        jnp.expand_dims(jnp.zeros_like(nt_states, dtype='bfloat16'), axis=2),
        gojax.get_action_size(states), axis=2)
    return losses.LossData(game.Trajectories(nt_states, nt_actions), nt_original_embeds=embeddings,
                           nt_curr_embeds=embeddings, nt_transition_logits=transition_logits,
                           nt_game_winners=jnp.zeros((batch_size, trajectory_length)))


class LossesTestCase(chex.TestCase):
    """Test losses.py"""

    def setUp(self):
        FLAGS.mark_as_parsed()

    def assertPytreeAllZero(self, pytree):
        # pylint: disable=invalid-name
        """Asserts all leaves in the pytree are zero."""
        if not functools.reduce(lambda a, b: a and b, map(lambda grad: (~grad.astype(bool)).all(),
                                                          jax.tree_util.tree_flatten(pytree)[0])):
            self.fail(f"PyTree has non-zero elements: {pytree}")

    def assertPytreeAllClose(self, pytree, expected_value, atol: Union[int, float] = 0,
                             rtol: Union[int, float] = 0):
        # pylint: disable=invalid-name
        """Asserts all leaves in the pytree are close to the expected value."""
        for array in jax.tree_util.tree_flatten(pytree)[0]:
            float_array = array.astype(float)
            np.testing.assert_allclose(float_array,
                                       jnp.full_like(float_array, fill_value=expected_value),
                                       atol=atol, rtol=rtol)

    def assertPytreeAnyNonZero(self, pytree):
        # pylint: disable=invalid-name
        """Asserts all leaves in the pytree are zero."""
        if not functools.reduce(lambda a, b: a or b, map(lambda grad: grad.astype(bool).any(),
                                                         jax.tree_util.tree_flatten(pytree)[0])):
            self.fail(f"PyTree no non-zero elements: {pytree}")

    def assertPytreeAllNonZero(self, pytree):
        # pylint: disable=invalid-name
        """Asserts all leaves in the pytree are non-zero."""
        if not functools.reduce(lambda a, b: a and b, map(lambda grad: grad.astype(bool).all(),
                                                          jax.tree_util.tree_flatten(pytree)[0])):
            self.fail(f"PyTree has zero elements: {pytree}")

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

    def test_assert_pytree_allclose(self):
        self.assertPytreeAllClose({'a': jnp.array(1e-7), 'b': {'c': jnp.array(1e-5)}}, 0, atol=1e-4)
        with self.assertRaises(AssertionError):
            self.assertPytreeAllClose({'a': jnp.array(0.5), 'b': {'c': jnp.array(1e-5)}}, 0,
                                      atol=1e-4)

    @flagsaver.flagsaver(board_size=3, hdim=2, embed_model='linear_conv', value_model='linear_conv',
                         policy_model='linear_conv', transition_model='linear_conv', hypo_steps=1,
                         add_value_loss=False, add_decode_loss=False, add_policy_loss=False,
                         add_trans_loss=False)
    def test_compute_loss_gradients_no_loss_no_gradients(self):
        """Tests all parameters except for transitions have grads with compute_0_step_total_loss."""
        go_model, params = models.make_model(FLAGS.board_size)
        params = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), params)
        trajectories = game.Trajectories(nt_states=jnp.ones((1, 1, 6, 3, 3), dtype=bool),
                                         nt_actions=jnp.ones((1, 1), dtype='uint16'))
        grads, _ = losses.compute_loss_gradients_and_metrics(go_model, params, trajectories)

        # Check all transition weights are non-zero.
        self.assertPytreeAllZero(grads)

    @flagsaver.flagsaver(board_size=3, hdim=2, embed_model='linear_conv', value_model='linear_conv',
                         policy_model='linear_conv', transition_model='linear_conv', hypo_steps=1,
                         add_value_loss=False, add_decode_loss=False, add_policy_loss=False,
                         add_trans_loss=True)
    def test_compute_loss_gradients_transition_loss_only_affects_transition_gradients(self):
        """Tests all parameters except for transitions have grads with compute_0_step_total_loss."""
        go_model, params = models.make_model(FLAGS.board_size)
        params = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), params)
        trajectories = game.Trajectories(nt_states=jnp.ones((1, 2, 6, 3, 3), dtype=bool),
                                         nt_actions=jnp.ones((1, 2), dtype='uint16'))
        grads: dict
        grads, _ = losses.compute_loss_gradients_and_metrics(go_model, params, trajectories)

        # Check a strict subset of transition weights are non-zero.
        self.assertPytreeAnyNonZero(grads['linear_conv_transition/~/conv2_d'])
        with self.assertRaises(AssertionError):
            self.assertPytreeAllNonZero(grads['linear_conv_transition/~/conv2_d'])
        # Check the remaining gradients are zero.
        grads.pop('linear_conv_transition/~/conv2_d')
        self.assertPytreeAllZero(grads)

    @flagsaver.flagsaver(board_size=3, hdim=2, embed_model='linear_conv', value_model='linear_conv',
                         policy_model='linear_conv', transition_model='linear_conv', hypo_steps=1,
                         add_value_loss=False, add_decode_loss=False, add_policy_loss=True,
                         add_trans_loss=False)
    def test_compute_loss_gradients_policy_loss_only_affects_embed_and_policy_gradients(self):
        """Tests all parameters except for transitions have grads with compute_0_step_total_loss."""
        go_model, params = models.make_model(FLAGS.board_size)
        params = jax.tree_util.tree_map(
            lambda x: jax.random.normal(jax.random.PRNGKey(42), x.shape, dtype='bfloat16'), params)
        trajectories = game.Trajectories(nt_states=jnp.ones((1, 2, 6, 3, 3), dtype=bool),
                                         nt_actions=jnp.ones((1, 2), dtype='uint16'))
        grads: dict
        grads, _ = losses.compute_loss_gradients_and_metrics(go_model, params, trajectories)

        # Check a strict subset of transition weights are non-zero.
        self.assertPytreeAllNonZero(grads['linear_conv_embed/~/conv2_d'])
        self.assertPytreeAllNonZero(grads['linear_conv_policy/~/conv2_d'])
        self.assertPytreeAllNonZero(grads['linear_conv_policy/~/conv2_d_1'])
        # Check the remaining gradients are zero.
        grads.pop('linear_conv_embed/~/conv2_d')
        grads.pop('linear_conv_policy/~/conv2_d')
        grads.pop('linear_conv_policy/~/conv2_d_1')
        self.assertPytreeAllZero(grads)

    @flagsaver.flagsaver(board_size=3, hdim=2, embed_model='linear_conv', value_model='linear_conv',
                         policy_model='linear_conv', transition_model='linear_conv', hypo_steps=1,
                         add_value_loss=False, add_decode_loss=False, add_policy_loss=True,
                         add_trans_loss=False)
    def test_compute_loss_gradients_policy_loss_zero_gradients_from_constant_trans_values(self):
        go_model, params = models.make_model(FLAGS.board_size)
        params = jax.tree_util.tree_map(
            lambda x: jax.random.normal(jax.random.PRNGKey(42), x.shape, dtype='bfloat16'), params)
        zero_value_params = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x),
                                                   params['linear_conv_value/~/conv2_d'])
        params['linear_conv_value/~/conv2_d'] = zero_value_params
        trajectories = game.Trajectories(nt_states=jnp.ones((1, 1, 6, 3, 3), dtype=bool),
                                         nt_actions=jnp.ones((1, 1), dtype='uint16'))
        grads: dict
        grads, _ = losses.compute_loss_gradients_and_metrics(go_model, params, trajectories)

        # Check a strict subset of transition weights are non-zero.
        self.assertPytreeAllZero(grads['linear_conv_embed/~/conv2_d'])
        self.assertPytreeAllZero(grads['linear_conv_policy/~/conv2_d'])
        self.assertPytreeAllZero(grads['linear_conv_policy/~/conv2_d_1'])

    @flagsaver.flagsaver(board_size=3, hdim=2, embed_model='linear_conv', value_model='linear_conv',
                         policy_model='linear_conv', transition_model='linear_conv', hypo_steps=1,
                         temperature=100, add_value_loss=False, add_decode_loss=False,
                         add_policy_loss=True, add_trans_loss=False)
    def test_compute_loss_gradients_policy_loss_zero_gradients_from_high_temperature(self):
        go_model, params = models.make_model(FLAGS.board_size)
        params = jax.tree_util.tree_map(
            lambda x: jax.random.normal(jax.random.PRNGKey(42), x.shape, dtype='bfloat16'), params)
        trajectories = game.Trajectories(nt_states=jnp.ones((1, 1, 6, 3, 3), dtype=bool),
                                         nt_actions=jnp.ones((1, 1), dtype='uint16'))
        grads: dict
        grads, _ = losses.compute_loss_gradients_and_metrics(go_model, params, trajectories)

        # Check a strict subset of transition weights are non-zero.
        self.assertPytreeAllClose(grads['linear_conv_embed/~/conv2_d'], 0, atol=1e-6)
        self.assertPytreeAllClose(grads['linear_conv_policy/~/conv2_d'], 0, atol=1e-6)
        self.assertPytreeAllClose(grads['linear_conv_policy/~/conv2_d_1'], 0, atol=1e-6)

    @flagsaver.flagsaver(board_size=3, hdim=2, embed_model='linear_conv', value_model='linear_conv',
                         policy_model='linear_conv', transition_model='linear_conv', hypo_steps=1,
                         temperature=0.01, add_value_loss=False, add_decode_loss=False,
                         add_policy_loss=True, add_trans_loss=False)
    def test_compute_loss_gradients_policy_loss_large_gradients_from_low_temperature(self):
        go_model, params = models.make_model(FLAGS.board_size)
        params = jax.tree_util.tree_map(
            lambda x: jax.random.normal(jax.random.PRNGKey(42), x.shape, dtype='bfloat16'), params)
        trajectories = game.Trajectories(nt_states=jnp.ones((1, 1, 6, 3, 3), dtype=bool),
                                         nt_actions=jnp.ones((1, 1), dtype='uint16'))
        grads: dict
        grads, _ = losses.compute_loss_gradients_and_metrics(go_model, params, trajectories)

        # Check a strict subset of transition weights are non-zero.
        self.assertPytreeAllNonZero(grads['linear_conv_embed/~/conv2_d'])
        self.assertPytreeAllNonZero(grads['linear_conv_policy/~/conv2_d'])
        self.assertPytreeAllNonZero(grads['linear_conv_policy/~/conv2_d_1'])

    @flagsaver.flagsaver(board_size=3, hdim=2, embed_model='linear_conv', value_model='linear_conv',
                         policy_model='linear_conv', transition_model='linear_conv', hypo_steps=1,
                         add_value_loss=True, add_decode_loss=False, add_policy_loss=False,
                         add_trans_loss=False)
    def test_compute_loss_gradients_value_loss_only_affects_embed_and_value_gradients(self):
        """Tests all parameters except for transitions have grads with compute_0_step_total_loss."""
        go_model, params = models.make_model(FLAGS.board_size)
        params = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), params)
        trajectories = game.Trajectories(nt_states=jnp.ones((1, 2, 6, 3, 3), dtype=bool),
                                         nt_actions=jnp.ones((1, 2), dtype='uint16'))
        grads: dict
        grads, _ = losses.compute_loss_gradients_and_metrics(go_model, params, trajectories)

        # Check a strict subset of transition weights are non-zero.
        self.assertPytreeAllNonZero(grads['linear_conv_embed/~/conv2_d'])
        self.assertPytreeAllNonZero(grads['linear_conv_value/~/conv2_d'])
        # Check the remaining gradients are zero.
        grads.pop('linear_conv_embed/~/conv2_d')
        grads.pop('linear_conv_value/~/conv2_d')
        self.assertPytreeAllZero(grads)

    @flagsaver.flagsaver(board_size=3, hdim=2, embed_model='linear_conv',
                         decode_model='linear_conv', value_model='linear_conv',
                         policy_model='linear_conv', transition_model='linear_conv', hypo_steps=1,
                         add_value_loss=False, add_decode_loss=True, add_policy_loss=False,
                         add_trans_loss=False)
    def test_compute_loss_gradients_decode_loss_only_affects_embed_and_decode_gradients(self):
        """Tests all parameters except for transitions have grads with compute_0_step_total_loss."""
        go_model, params = models.make_model(FLAGS.board_size)
        params = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), params)
        params['linear_conv_decode/~/conv2_d'] = jax.tree_util.tree_map(
            lambda x: -1 * jnp.ones_like(x), params['linear_conv_decode/~/conv2_d'])
        trajectories = game.Trajectories(nt_states=jnp.ones((1, 2, 6, 3, 3), dtype=bool),
                                         nt_actions=jnp.ones((1, 2), dtype='uint16'))
        grads: dict
        grads, _ = losses.compute_loss_gradients_and_metrics(go_model, params, trajectories)

        # Check a strict subset of transition weights are non-zero.
        self.assertPytreeAllNonZero(grads['linear_conv_embed/~/conv2_d'])
        self.assertPytreeAllNonZero(grads['linear_conv_decode/~/conv2_d'])
        # Check the remaining gradients are zero.
        grads.pop('linear_conv_embed/~/conv2_d')
        grads.pop('linear_conv_decode/~/conv2_d')
        self.assertPytreeAllZero(grads)

    @flagsaver.flagsaver(board_size=3, hdim=2, embed_model='linear_conv', value_model='linear',
                         policy_model='linear', transition_model='linear_conv', hypo_steps=2,
                         add_trans_loss=True)
    def test_compute_loss_gradients_with_two_steps_and_trans_loss_has_nonzero_grads(self):
        """Tests all parameters except for transitions have grads with compute_0_step_total_loss."""
        go_model, params = models.make_model(FLAGS.board_size)
        trajectories = game.Trajectories(nt_states=jnp.ones((1, 2, 6, 3, 3), dtype=bool),
                                         nt_actions=jnp.ones((1, 2), dtype='uint16'))
        grads, _ = losses.compute_loss_gradients_and_metrics(go_model, params, trajectories)

        # Check some transition weights are non-zero.
        self.assertTrue(grads['linear_conv_transition/~/conv2_d']['b'].astype(bool).any())
        self.assertTrue(grads['linear_conv_transition/~/conv2_d']['w'].astype(bool).any())
        # Check everything else is non-zero.
        del grads['linear_conv_transition/~/conv2_d']['b']
        del grads['linear_conv_transition/~/conv2_d']['w']
        self.assertTrue(functools.reduce(lambda a, b: a and b,
                                         map(lambda grad: grad.astype(bool).all(),
                                             jax.tree_util.tree_flatten(grads)[0])))

    @flagsaver.flagsaver(board_size=3, hdim=2, embed_model='linear_conv', value_model='linear',
                         policy_model='linear', transition_model='linear_conv', hypo_steps=2,
                         add_trans_loss=False)
    def test_compute_loss_gradients_with_two_steps_and_no_trans_loss_has_no_trans_grads(self):
        """Tests all parameters except for transitions have grads with compute_0_step_total_loss."""
        go_model, params = models.make_model(FLAGS.board_size)
        trajectories = game.Trajectories(nt_states=jnp.ones((1, 3, 6, 3, 3), dtype=bool),
                                         nt_actions=jnp.ones((1, 3), dtype='uint16'))
        grads, _ = losses.compute_loss_gradients_and_metrics(go_model, params, trajectories)

        # Check some transition weights are non-zero.
        self.assertFalse(grads['linear_conv_transition/~/conv2_d']['b'].astype(bool).any())
        self.assertFalse(grads['linear_conv_transition/~/conv2_d']['w'].astype(bool).any())


if __name__ == '__main__':
    absltest.main()
