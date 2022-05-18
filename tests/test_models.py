"""Tests model.py."""
# pylint: disable=missing-function-docstring,no-self-use,unnecessary-lambda
import unittest

import chex
import gojax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

import models


class MockModelTestCase(chex.TestCase):
    """Tests the mock models work correctly."""

    @parameterized.named_parameters(
        ('scalars', 0, 1, 2, 3),
        ('arrays', [0], [[1]], [[2, 3], [4, 5]], 6)
    )
    def test_outputs_(self, state_embed_output, value_output, policy_output, transition_output):
        mock_model_fn = models.make_mock_model(state_embed_output, value_output, policy_output,
                                               transition_output)
        for batch_size in (1, 2):
            new_states = gojax.new_states(board_size=3, batch_size=batch_size)
            params = mock_model_fn.init(jax.random.PRNGKey(42), new_states)
            self.assertIsInstance(params, dict)
            self.assertEmpty(params)
            rng = None
            state_embed_model, value_model, policy_model, transition_model = mock_model_fn.apply
            np.testing.assert_array_equal(state_embed_model(params, rng, new_states),
                                          jnp.repeat(jnp.array([state_embed_output]), batch_size,
                                                     axis=0))
            np.testing.assert_array_equal(value_model(params, rng, new_states),
                                          jnp.repeat(jnp.array([value_output]), batch_size, axis=0))
            np.testing.assert_array_equal(policy_model(params, rng, new_states),
                                          jnp.repeat(jnp.array([policy_output]), batch_size,
                                                     axis=0))
            np.testing.assert_array_equal(transition_model(params, rng, new_states),
                                          jnp.repeat(jnp.array([transition_output]), batch_size,
                                                     axis=0))


class ModelOutputShapeTestCase(chex.TestCase):
    """Tests output shape of various model architectures."""

    @parameterized.named_parameters(
        ('_random', 'identity', 'random', 'random', 'random', (1, gojax.NUM_CHANNELS, 3, 3), (1,),
         (1, 10), (1, 10, gojax.NUM_CHANNELS, 3, 3)),
        ('_linear', 'identity', 'linear', 'linear', 'linear', (1, gojax.NUM_CHANNELS, 3, 3), (1,),
         (1, 10), (1, 10, gojax.NUM_CHANNELS, 3, 3)),
    )
    def test_single_batch_board_size_three(self, state_embed_model_name, value_model_name,
                                           policy_model_name,
                                           transition_model_name, expected_embed_shape,
                                           expected_value_shape,
                                           expected_policy_shape, expected_transition_shape):
        # pylint: disable=too-many-arguments
        # Build the model
        board_size = 3
        model_fn = models.make_model(board_size, state_embed_model_name, value_model_name,
                                     policy_model_name,
                                     transition_model_name)
        new_states = gojax.new_states(batch_size=1, board_size=board_size)
        params = model_fn.init(jax.random.PRNGKey(42), new_states)
        # Check the shapes
        chex.assert_shape((model_fn.apply[0](params, jax.random.PRNGKey(42), new_states),
                           model_fn.apply[1](params, jax.random.PRNGKey(42), new_states),
                           model_fn.apply[2](params, jax.random.PRNGKey(42), new_states),
                           model_fn.apply[3](params, jax.random.PRNGKey(42), new_states)),
                          (expected_embed_shape, expected_value_shape, expected_policy_shape,
                           expected_transition_shape))


class ModelTestCase(chex.TestCase):
    """Tests model.py."""

    def test_get_random_model_params(self):
        board_size = 3
        model_fn = models.make_model(board_size, 'identity', 'random', 'random', 'random')
        self.assertIsInstance(model_fn, hk.MultiTransformed)
        params = model_fn.init(jax.random.PRNGKey(42),
                               gojax.new_states(batch_size=2, board_size=board_size))
        self.assertIsInstance(params, dict)
        self.assertEqual(len(params), 0)

    def test_get_linear_model_params(self):
        board_size = 3
        model_fn = models.make_model(board_size, 'identity', 'linear', 'linear', 'linear')
        self.assertIsInstance(model_fn, hk.MultiTransformed)
        params = model_fn.init(jax.random.PRNGKey(42),
                               gojax.new_states(batch_size=2, board_size=board_size))
        self.assertIsInstance(params, dict)
        chex.assert_tree_all_equal_structs(params, {'linear3_d_policy': {'action_w': 0},
                                                    'linear3_d_transition': {'transition_b': 0,
                                                                             'transition_w': 0},
                                                    'linear3_d_value': {'value_b': 0,
                                                                        'value_w': 0}})

    def test_get_linear_model_output_zero_params(self):
        board_size = 3
        model_fn = hk.without_apply_rng(
            models.make_model(board_size, 'identity', 'linear', 'linear', 'linear'))
        new_states = gojax.new_states(batch_size=1, board_size=board_size)
        params = model_fn.init(jax.random.PRNGKey(42), new_states)
        params = jax.tree_map(lambda p: jnp.zeros_like(p), params)

        ones_like_states = jnp.ones_like(new_states)
        state_embed_model = model_fn.apply[0]
        output = state_embed_model(params, ones_like_states)
        np.testing.assert_array_equal(output, ones_like_states)

        for sub_model in model_fn.apply[1:]:
            output = sub_model(params, ones_like_states)
            np.testing.assert_array_equal(output, jnp.zeros_like(output))

    def test_get_linear_model_output_ones_params(self):
        board_size = 3
        model_fn = hk.without_apply_rng(
            models.make_model(board_size, 'identity', 'linear', 'linear', 'linear'))
        new_states = gojax.new_states(batch_size=1, board_size=board_size)
        params = model_fn.init(jax.random.PRNGKey(42), new_states)
        params = jax.tree_map(lambda p: jnp.ones_like(p), params)

        ones_like_states = jnp.ones_like(new_states)
        state_embed_model, value_model, policy_model, transition_model = model_fn.apply
        output = state_embed_model(params, ones_like_states)
        np.testing.assert_array_equal(output, ones_like_states)

        value_output = value_model(params, ones_like_states)
        np.testing.assert_array_equal(value_output,
                                      jnp.full_like(value_output,
                                                    gojax.NUM_CHANNELS * board_size ** 2
                                                    + 1))
        policy_output = policy_model(params, ones_like_states)
        np.testing.assert_array_equal(policy_output,
                                      jnp.full_like(policy_output,
                                                    gojax.NUM_CHANNELS * board_size ** 2))
        transition_output = transition_model(params, ones_like_states)
        np.testing.assert_array_equal(transition_output,
                                      jnp.full_like(transition_output,
                                                    gojax.NUM_CHANNELS * board_size ** 2
                                                    + 1))

    def test_get_real_transition_model_output(self):
        board_size = 3
        model_fn = hk.without_apply_rng(
            models.make_model(board_size, 'identity', 'linear', 'linear', 'real'))
        new_states = gojax.new_states(batch_size=1, board_size=board_size)
        params = model_fn.init(jax.random.PRNGKey(42), new_states)

        transition_model = model_fn.apply[3]
        transition_output = transition_model(params, new_states)
        expected_transition = jnp.expand_dims(
            jnp.concatenate((gojax.decode_states("""
                                                  B _ _
                                                  _ _ _
                                                  _ _ _
                                                  """,
                                                 turn=gojax.WHITES_TURN),
                             gojax.decode_states("""
                                                  _ B _
                                                  _ _ _
                                                  _ _ _
                                                  """,
                                                 turn=gojax.WHITES_TURN),
                             gojax.decode_states("""
                                                  _ _ B
                                                  _ _ _
                                                  _ _ _
                                                  """,
                                                 turn=gojax.WHITES_TURN),
                             gojax.decode_states("""
                                                  _ _ _
                                                  B _ _
                                                  _ _ _
                                                  """,
                                                 turn=gojax.WHITES_TURN),
                             gojax.decode_states("""
                                                  _ _ _
                                                  _ B _
                                                  _ _ _
                                                  """,
                                                 turn=gojax.WHITES_TURN),
                             gojax.decode_states("""
                                                  _ _ _
                                                  _ _ B
                                                  _ _ _
                                                  """,
                                                 turn=gojax.WHITES_TURN),
                             gojax.decode_states("""
                                                  _ _ _
                                                  _ _ _
                                                  B _ _
                                                  """,
                                                 turn=gojax.WHITES_TURN),
                             gojax.decode_states("""
                                                  _ _ _
                                                  _ _ _
                                                  _ B _
                                                  """,
                                                 turn=gojax.WHITES_TURN),
                             gojax.decode_states("""
                                                  _ _ _
                                                  _ _ _
                                                  _ _ B
                                                  """,
                                                 turn=gojax.WHITES_TURN),
                             gojax.decode_states("""
                                                  _ _ _
                                                  _ _ _
                                                  _ _ _
                                                  """,
                                                 turn=gojax.WHITES_TURN,
                                                 passed=True),
                             )
                            ), axis=0)
        np.testing.assert_array_equal(transition_output, expected_transition)

    def test_get_unknown_model(self):
        board_size = 3
        with self.assertRaises(KeyError):
            models.make_model(board_size, 'foo', 'foo', 'foo', 'foo')

    if __name__ == '__main__':
        unittest.main()
