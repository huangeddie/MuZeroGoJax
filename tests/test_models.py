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


class ModelOutputShapeTestCase(chex.TestCase):
    """Tests output shape of various model architectures."""

    @parameterized.named_parameters(
        ('_random', 'identity', 'random', 'random', 'random', (1, gojax.NUM_CHANNELS, 3, 3), (1,),
         (1, 10), (1, 10, gojax.NUM_CHANNELS, 3, 3)),
        ('_linear', 'identity', 'linear', 'linear', 'linear', (1, gojax.NUM_CHANNELS, 3, 3), (1,),
         (1, 10), (1, 10, gojax.NUM_CHANNELS, 3, 3)),
    )
    def test_single_batch_board_size_three(self, embed_model_name, value_model_name,
                                           policy_model_name,
                                           transition_model_name, expected_embed_shape,
                                           expected_value_shape,
                                           expected_policy_shape, expected_transition_shape):
        # pylint: disable=too-many-arguments
        # Build the model
        board_size = 3
        model_fn = models.make_model(board_size, embed_model_name, value_model_name,
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


class EmbedModelTestCase(chex.TestCase):
    """Tests embed models."""

    @parameterized.named_parameters(
        ('black_turn_noop',
         gojax.decode_states("""
        _ _ _
        _ _ _
        _ _ _
        """),
         gojax.decode_states("""
        _ _ _
        _ _ _
        _ _ _
        """)),
        ('white_to_black',
         gojax.decode_states("""
            _ _ _
            _ _ _
            _ _ _
            TURN=W
            """),
         gojax.decode_states("""
            _ _ _
            _ _ _
            _ _ _
            TURN=B
            """)),
        ('white_to_black_with_pieces',
         gojax.decode_states("""
                _ _ _
                _ B _
                _ W _
                TURN=W
                """),
         gojax.decode_states("""
                _ _ _
                _ W _
                _ B _
                TURN=B
                """)),
        ('black_and_white_to_black_and_black_with_pieces',
         gojax.decode_states("""
                    B _ _
                    W _ _
                    _ _ _
                    TURN=B
                    
                    _ _ _
                    _ B _
                    _ W _
                    TURN=W
                    """),
         gojax.decode_states("""
                    B _ _
                    W _ _
                    _ _ _
                    TURN=B
         
                    _ _ _
                    _ W _
                    _ B _
                    TURN=B
                    """)),
    )
    def test_black_perspective_(self, states, expected_embedding):
        embed_model = hk.without_apply_rng(
            hk.transform(lambda x: models.embed.BlackPerspective(board_size=3)(x)))
        rng = jax.random.PRNGKey(42)
        params = embed_model.init(rng, states)
        self.assertEmpty(params)
        np.testing.assert_array_equal(embed_model.apply(params, states), expected_embedding)


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
