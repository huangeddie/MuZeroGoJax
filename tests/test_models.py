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

from muzero_gojax import main
from muzero_gojax import models


class OutputShapeTestCase(chex.TestCase):
    """Tests the output shape of models."""

    @parameterized.named_parameters(('black_cnn_lite', models.embed.BlackCNNLite, 32, (2, 32, 3, 3)),
                                    ('cnn_intermediate', models.embed.CNNIntermediateEmbed, 256, (2, 256, 3, 3)),
                                    ('black_cnn_intermediate', models.embed.BlackCNNIntermediate, 256, (2, 256, 3, 3)),
                                    ('cnn_lite_policy', models.policy.CNNLitePolicy, 6, (2, 10)), )
    def test_from_two_states_(self, model_class, hdim, expected_shape):
        board_size = 3
        model = hk.without_apply_rng(hk.transform(lambda x: model_class(board_size, hdim)(x)))
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
            hk.transform(lambda x: models.embed.BlackPerspective(board_size=3, hdim=None)(x)))
        rng = jax.random.PRNGKey(42)
        params = embed_model.init(rng, states)
        self.assertEmpty(params)
        np.testing.assert_array_equal(embed_model.apply(params, states), expected_embedding)


class TransitionTestCase(chex.TestCase):
    """Tests the transition models."""

    def test_get_real_transition_model_output(self):
        board_size = 3
        main.FLAGS(f'foo --board_size={board_size} --embed_model=identity --value_model=linear '
                   '--policy_model=linear --transition_model=real'.split())
        go_model = hk.without_apply_rng(models.make_model(main.FLAGS))
        params = go_model.init(jax.random.PRNGKey(42),
                               states=jnp.zeros((1, gojax.NUM_CHANNELS, board_size, board_size)))

        transition_model = go_model.apply[3]
        transition_output = transition_model(params,
                                             embeds=jnp.zeros((1, gojax.NUM_CHANNELS + 1, board_size, board_size)))
        expected_transition = gojax.decode_states("""
                                                  _ _ _
                                                  _ _ _
                                                  _ _ _
                                                  PASS=T
                                                  """, turn=gojax.WHITES_TURN)
        np.testing.assert_array_equal(transition_output, expected_transition)

    def test_get_cnn_intermediate_transition_model_output_shape(self):
        board_size = 3
        hdim = 4
        main.FLAGS(f'foo --board_size={board_size} --embed_model=identity --value_model=linear '
                   f'--policy_model=linear --transition_model=cnn_intermediate --hdim={hdim}'.split())
        go_model = hk.without_apply_rng(models.make_model(main.FLAGS))
        params = go_model.init(jax.random.PRNGKey(42),
                               states=jnp.zeros((1, gojax.NUM_CHANNELS, board_size, board_size)))

        transition_model = go_model.apply[3]
        chex.assert_shape(
            transition_model(params, embeds=jnp.zeros((1, gojax.NUM_CHANNELS + 1, board_size, board_size))),
            (1, hdim, board_size, board_size))

    def test_get_cnn_lite_transition_model_output_shape(self):
        board_size = 3
        hdim = 4
        main.FLAGS(f'foo --board_size={board_size} --embed_model=identity --value_model=linear '
                   f'--policy_model=linear --transition_model=cnn_lite --hdim={hdim}'.split())
        go_model = hk.without_apply_rng(models.make_model(main.FLAGS))
        params = go_model.init(jax.random.PRNGKey(42),
                               states=jnp.zeros((1, gojax.NUM_CHANNELS, board_size, board_size)))

        transition_model = go_model.apply[3]
        chex.assert_shape(
            transition_model(params, embeds=jnp.zeros((1, gojax.NUM_CHANNELS + 1, board_size, board_size))),
            (1, hdim, board_size, board_size))


class ValueTestCase(chex.TestCase):
    """Tests the value models."""

    def test_tromp_taylor_value_model_output(self):
        states = gojax.decode_states("""
                                    _ B B
                                    _ W _
                                    _ _ _
                                    TURN=B
                                    
                                    _ W _
                                    _ _ _
                                    _ _ _
                                    TURN=W
                                    """)
        tromp_taylor_value = hk.without_apply_rng(
            hk.transform(lambda x: models.value.TrompTaylorValue(board_size=3, hdim=None)(x)))
        params = tromp_taylor_value.init(None, states)
        self.assertEmpty(params)
        np.testing.assert_array_equal(tromp_taylor_value.apply(params, states), [1, 9])


class PolicyTestCase(chex.TestCase):
    """Tests the policy models."""

    def test_tromp_taylor_policy_model_output(self):
        states = gojax.decode_states("""
                                    _ B B
                                    _ W _
                                    _ _ _
                                    TURN=B

                                    _ W _
                                    _ _ _
                                    _ _ _
                                    TURN=W
                                    """)
        tromp_taylor_policy = hk.without_apply_rng(
            hk.transform(lambda x: models.policy.TrompTaylorPolicy(board_size=3, hdim=None)(x)))
        params = tromp_taylor_policy.init(None, states)
        self.assertEmpty(params)
        np.testing.assert_array_equal(tromp_taylor_policy.apply(params, states),
                                      [[2, 1, 1, 3, 1, 2, 2, 2, 2, 1], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]])


class MakeModelTestCase(chex.TestCase):
    """Tests model.py."""

    def test_get_random_model_params(self):
        board_size = 3
        main.FLAGS(f'foo --board_size={board_size} --embed_model=identity --value_model=random '
                   '--policy_model=random --transition_model=random'.split())
        go_model = models.make_model(main.FLAGS)
        self.assertIsInstance(go_model, hk.MultiTransformed)
        params = go_model.init(jax.random.PRNGKey(42), gojax.new_states(batch_size=2, board_size=board_size))
        self.assertIsInstance(params, dict)
        self.assertEqual(len(params), 0)

    def test_get_linear_model_params(self):
        board_size = 3
        main.FLAGS(f'foo --board_size={board_size} --embed_model=identity --value_model=linear '
                   '--policy_model=linear --transition_model=linear'.split())
        go_model = models.make_model(main.FLAGS)
        self.assertIsInstance(go_model, hk.MultiTransformed)
        params = go_model.init(jax.random.PRNGKey(42), gojax.new_states(batch_size=2, board_size=board_size))
        self.assertIsInstance(params, dict)
        chex.assert_tree_all_equal_structs(params, {'linear3_d_policy': {'action_w': 0},
                                                    'linear3_d_transition': {'transition_b': 0, 'transition_w': 0},
                                                    'linear3_d_value': {'value_b': 0, 'value_w': 0}})

    def test_get_linear_model_output_ones_params(self):
        board_size = 3
        main.FLAGS(f'foo --board_size={board_size} --embed_model=identity --value_model=linear '
                   '--policy_model=linear --transition_model=linear'.split())
        go_model = hk.without_apply_rng(models.make_model(main.FLAGS))
        new_states = gojax.new_states(batch_size=1, board_size=board_size)
        params = go_model.init(jax.random.PRNGKey(42), new_states)
        params = jax.tree_map(lambda p: jnp.ones_like(p), params)

        ones_like_states = jnp.ones_like(new_states)
        embed_model, value_model, policy_model, transition_model = go_model.apply
        output = embed_model(params, ones_like_states)
        np.testing.assert_array_equal(output, ones_like_states)

        value_output = value_model(params, ones_like_states)
        np.testing.assert_array_equal(value_output,
                                      jnp.full_like(value_output, gojax.NUM_CHANNELS * board_size ** 2 + 1))
        policy_output = policy_model(params, ones_like_states)
        np.testing.assert_array_equal(policy_output, jnp.full_like(policy_output, gojax.NUM_CHANNELS * board_size ** 2))
        transition_output = transition_model(params, jnp.concatenate(
            (ones_like_states, jnp.ones((1, 1, board_size, board_size))), axis=1))
        np.testing.assert_array_equal(transition_output,
                                      jnp.full_like(transition_output, (gojax.NUM_CHANNELS + 1) * board_size ** 2 + 1))

    if __name__ == '__main__':
        unittest.main()
