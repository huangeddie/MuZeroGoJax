"""Tests model.py."""
# pylint: disable=missing-function-docstring,no-self-use,unnecessary-lambda,duplicate-code
import unittest

import chex
import gojax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from muzero_gojax import models


class OutputShapeTestCase(chex.TestCase):
    """Tests the output shape of models."""

    @parameterized.named_parameters(('black_cnn_lite', models.embed.BlackCNNLite, (2, 32, 3, 3)), (
            'black_cnn_intermediate', models.embed.BlackCNNIntermediate, (2, 256, 3, 3)), (
                                            'black_real_perspective',
                                            models.transition.BlackRealTransition,
                                            (2, 10, gojax.NUM_CHANNELS, 3, 3)), (
                                            'cnn_lite_transition',
                                            models.transition.CNNLiteTransition, (2, 10, 32, 3, 3)),
                                    ('cnn_intermediate_transition',
                                     models.transition.CNNIntermediateTransition,
                                     (2, 10, 256, 3, 3)),
                                    ('cnn_lite_policy', models.policy.CNNLitePolicy, (2, 10)), )
    def test_from_two_states_(self, model_class, expected_shape):
        board_size = 3
        model = hk.without_apply_rng(hk.transform(lambda x: model_class(board_size)(x)))
        states = gojax.new_states(batch_size=2, board_size=board_size)
        params = model.init(jax.random.PRNGKey(42), states)
        output = model.apply(params, states)
        chex.assert_shape(output, expected_shape)


class EmbedModelTestCase(chex.TestCase):
    """Tests embed models."""

    def test_black_perspective(self):
        states = gojax.decode_states("""
                    B _ _
                    W _ _
                    _ _ _
                    TURN=B
                    
                    _ _ _
                    _ B _
                    _ W _
                    TURN=W
                    """)
        expected_embedding = gojax.decode_states("""
                    B _ _
                    W _ _
                    _ _ _
                    TURN=B
         
                    _ _ _
                    _ W _
                    _ B _
                    TURN=B
                    """)
        embed_model = hk.without_apply_rng(
            hk.transform(lambda x: models.embed.BlackPerspective(board_size=3)(x)))
        rng = jax.random.PRNGKey(42)
        params = embed_model.init(rng, states)
        self.assertEmpty(params)
        np.testing.assert_array_equal(embed_model.apply(params, states), expected_embedding)


class TransitionTestCase(chex.TestCase):
    """Tests the transition models."""

    def test_get_real_transition_model_output(self):
        board_size = 3
        model_fn = hk.without_apply_rng(
            models.make_model(board_size, 'identity', 'linear', 'linear', 'real'))
        new_states = gojax.new_states(batch_size=1, board_size=board_size)
        params = model_fn.init(jax.random.PRNGKey(42), new_states)

        transition_model = model_fn.apply[3]
        transition_output = transition_model(params, new_states)
        expected_transition = jnp.expand_dims(gojax.decode_states("""
                              B _ _
                              _ _ _
                              _ _ _

                              _ B _
                              _ _ _
                              _ _ _

                              _ _ B
                              _ _ _
                              _ _ _

                              _ _ _
                              B _ _
                              _ _ _

                              _ _ _
                              _ B _
                              _ _ _

                              _ _ _
                              _ _ B
                              _ _ _

                              _ _ _
                              _ _ _
                              B _ _

                              _ _ _
                              _ _ _
                              _ B _

                              _ _ _
                              _ _ _
                              _ _ B
                              
                              _ _ _
                              _ _ _
                              _ _ _
                              PASS=T
                              """, turn=gojax.WHITES_TURN), axis=0)
        np.testing.assert_array_equal(transition_output, expected_transition)


class MakeModelTestCase(chex.TestCase):
    """Tests model.py."""

    @parameterized.named_parameters((
            '_random', 'identity', 'random', 'random', 'random', (1, gojax.NUM_CHANNELS, 3, 3),
            (1,), (1, 10), (1, 10, gojax.NUM_CHANNELS, 3, 3)), (
            '_linear', 'identity', 'linear', 'linear', 'linear', (1, gojax.NUM_CHANNELS, 3, 3),
            (1,), (1, 10), (1, 10, gojax.NUM_CHANNELS, 3, 3)), )
    def test_single_batch_board_size_three(self, embed_model_name, value_model_name,
                                           policy_model_name, transition_model_name,
                                           expected_embed_shape, expected_value_shape,
                                           expected_policy_shape, expected_transition_shape):
        # pylint: disable=too-many-arguments
        # Build the model
        board_size = 3
        model_fn = models.make_model(board_size, embed_model_name, value_model_name,
                                     policy_model_name, transition_model_name)
        new_states = gojax.new_states(batch_size=1, board_size=board_size)
        params = model_fn.init(jax.random.PRNGKey(42), new_states)
        # Check the shapes
        chex.assert_shape((model_fn.apply[0](params, jax.random.PRNGKey(42), new_states),
                           model_fn.apply[1](params, jax.random.PRNGKey(42), new_states),
                           model_fn.apply[2](params, jax.random.PRNGKey(42), new_states),
                           model_fn.apply[3](params, jax.random.PRNGKey(42), new_states)), (
                              expected_embed_shape, expected_value_shape, expected_policy_shape,
                              expected_transition_shape))

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
        embed_model = model_fn.apply[0]
        output = embed_model(params, ones_like_states)
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
        embed_model, value_model, policy_model, transition_model = model_fn.apply
        output = embed_model(params, ones_like_states)
        np.testing.assert_array_equal(output, ones_like_states)

        value_output = value_model(params, ones_like_states)
        np.testing.assert_array_equal(value_output, jnp.full_like(value_output,
                                                                  gojax.NUM_CHANNELS * board_size
                                                                  ** 2 + 1))
        policy_output = policy_model(params, ones_like_states)
        np.testing.assert_array_equal(policy_output, jnp.full_like(policy_output,
                                                                   gojax.NUM_CHANNELS *
                                                                   board_size ** 2))
        transition_output = transition_model(params, ones_like_states)
        np.testing.assert_array_equal(transition_output, jnp.full_like(transition_output,
                                                                       gojax.NUM_CHANNELS *
                                                                       board_size ** 2 + 1))

    def test_get_unknown_model(self):
        board_size = 3
        with self.assertRaises(KeyError):
            models.make_model(board_size, 'foo', 'foo', 'foo', 'foo')

    if __name__ == '__main__':
        unittest.main()
