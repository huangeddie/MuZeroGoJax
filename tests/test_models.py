"""Tests model.py."""
# pylint: disable=missing-function-docstring,duplicate-code
import chex
import gojax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized

from muzero_gojax import main
from muzero_gojax import models
from muzero_gojax.models import decode
from muzero_gojax.models import embed
from muzero_gojax.models import policy
from muzero_gojax.models import transition
from muzero_gojax.models import value

FLAGS = main.FLAGS


class ModelsTestCase(chex.TestCase):
    """Tests models.py."""

    def setUp(self):
        FLAGS.mark_as_parsed()

    @parameterized.named_parameters(  # Embed
        (embed.Identity.__name__, embed.Identity, 6, (2, 6, 3, 3)),
        (embed.BlackPerspective.__name__, embed.BlackPerspective, 6, (2, 6, 3, 3)),
        (embed.BlackCnnLite.__name__, embed.BlackCnnLite, 6, (2, 6, 3, 3)),
        (embed.LinearConvEmbed.__name__, embed.LinearConvEmbed, 2, (2, 2, 3, 3)),
        (embed.CnnLiteEmbed.__name__, embed.CnnLiteEmbed, 2, (2, 2, 3, 3)),
        (embed.ResNetV2Embed.__name__, embed.ResNetV2Embed, 2, (2, 2, 3, 3)),  # Decode
        (decode.NoOpDecode.__name__, decode.NoOpDecode, 2, (2, 6, 3, 3)),
        (decode.LinearConvDecode.__name__, decode.LinearConvDecode, 2, (2, 6, 3, 3)),
        (decode.ResNetV2Decode.__name__, decode.ResNetV2Decode, 2, (2, 6, 3, 3)),  # Value
        (value.RandomValue.__name__, value.RandomValue, 2, (2,)),
        (value.LinearConvValue.__name__, value.LinearConvValue, 2, (2,)),
        (value.Linear3DValue.__name__, value.Linear3DValue, 2, (2,)),
        (value.CnnLiteValue.__name__, value.CnnLiteValue, 2, (2,)),
        (value.ResnetMediumValue.__name__, value.ResnetMediumValue, 2, (2,)),
        (value.TrompTaylorValue.__name__, value.TrompTaylorValue, 2, (2,)),  # Policy
        (policy.RandomPolicy.__name__, policy.RandomPolicy, 2, (2, 10)),
        (policy.Linear3DPolicy.__name__, policy.Linear3DPolicy, 2, (2, 10)),
        (policy.LinearConvPolicy.__name__, policy.LinearConvPolicy, 2, (2, 10)),
        (policy.CnnLitePolicy.__name__, policy.CnnLitePolicy, 2, (2, 10)),
        (policy.ResnetMediumPolicy.__name__, policy.ResnetMediumPolicy, 2, (2, 10)),
        (policy.TrompTaylorPolicy.__name__, policy.TrompTaylorPolicy, 2, (2, 10)),  # Transition
        (transition.RealTransition.__name__, transition.RealTransition, 6, (2, 10, 6, 3, 3)), (
                transition.BlackRealTransition.__name__, transition.BlackRealTransition, 6,
                (2, 10, 6, 3, 3)),
        (transition.RandomTransition.__name__, transition.RandomTransition, 2, (2, 10, 2, 3, 3)), (
                transition.LinearConvTransition.__name__, transition.LinearConvTransition, 2,
                (2, 10, 2, 3, 3)),
        (transition.CnnLiteTransition.__name__, transition.CnnLiteTransition, 2, (2, 10, 2, 3, 3)),
        (transition.ResnetMediumTransition.__name__, transition.ResnetMediumTransition, 2,
         (2, 10, 2, 3, 3)), (
                transition.ResNetV2Transition.__name__, transition.ResNetV2Transition, 2,
                (2, 10, 2, 3, 3)))
    def test_model_output(self, model_class, embed_dim, expected_shape):
        with flagsaver.flagsaver(board_size=3, hdim=4, embed_dim=embed_dim):
            model = hk.transform(lambda x: model_class(FLAGS)(x))
            states = gojax.new_states(batch_size=2, board_size=3)
            params = model.init(jax.random.PRNGKey(42), states)
            output = model.apply(params, jax.random.PRNGKey(42), states)
            chex.assert_shape(output, expected_shape)
            chex.assert_type(output, 'bfloat16')

    @flagsaver.flagsaver(board_size=3, hdim=4)
    def test_embed_black_perspective(self):
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
        embed_model = hk.without_apply_rng(hk.transform(lambda x: embed.BlackPerspective(FLAGS)(x)))
        rng = jax.random.PRNGKey(42)
        params = embed_model.init(rng, states)
        self.assertEmpty(params)
        np.testing.assert_array_equal(embed_model.apply(params, states), expected_embedding)

    @flagsaver.flagsaver(board_size=3, embed_model='identity', value_model='linear',
                         policy_model='linear', transition_model='real')
    def test_get_real_transition_model_output(self):
        go_model = hk.without_apply_rng(models.make_model(FLAGS.board_size))
        new_states = gojax.new_states(batch_size=1, board_size=3)
        params = go_model.init(jax.random.PRNGKey(42), new_states)

        transition_model = go_model.apply[models.TRANSITION_INDEX]
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

    @flagsaver.flagsaver(board_size=3, embed_model='identity', value_model='linear',
                         policy_model='linear', transition_model='black_perspective')
    def test_transition_black_perspective_output(self):
        go_model = hk.without_apply_rng(models.make_model(FLAGS.board_size))
        new_states = gojax.new_states(batch_size=1, board_size=3)
        params = go_model.init(jax.random.PRNGKey(42), new_states)

        transition_model = go_model.apply[models.TRANSITION_INDEX]
        transition_output = transition_model(params, new_states)
        expected_transition = jnp.expand_dims(gojax.decode_states("""
                              W _ _
                              _ _ _
                              _ _ _

                              _ W _
                              _ _ _
                              _ _ _

                              _ _ W
                              _ _ _
                              _ _ _

                              _ _ _
                              W _ _
                              _ _ _

                              _ _ _
                              _ W _
                              _ _ _

                              _ _ _
                              _ _ W
                              _ _ _

                              _ _ _
                              _ _ _
                              W _ _

                              _ _ _
                              _ _ _
                              _ W _

                              _ _ _
                              _ _ _
                              _ _ W

                              _ _ _
                              _ _ _
                              _ _ _
                              PASS=T
                              """), axis=0)
        np.testing.assert_array_equal(transition_output, expected_transition)

    @flagsaver.flagsaver(board_size=3, hdim=4, embed_dim=2)
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
            hk.transform(lambda x: models.value.TrompTaylorValue(FLAGS)(x)))
        params = tromp_taylor_value.init(None, states)
        self.assertEmpty(params)
        np.testing.assert_array_equal(tromp_taylor_value.apply(params, states), [1, 9])

    @flagsaver.flagsaver(board_size=3, hdim=4, embed_dim=2)
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
            hk.transform(lambda x: models.policy.TrompTaylorPolicy(FLAGS)(x)))
        params = tromp_taylor_policy.init(None, states)
        self.assertEmpty(params)
        np.testing.assert_array_equal(tromp_taylor_policy.apply(params, states),
                                      [[2, 1, 1, 3, 1, 2, 2, 2, 2, 1],
                                       [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]])

    @flagsaver.flagsaver(board_size=3, embed_model='identity', value_model='random',
                         policy_model='random', transition_model='random')
    def test_make_random_model_params(self):
        go_model = models.make_model(FLAGS.board_size)
        self.assertIsInstance(go_model, hk.MultiTransformed)
        params = go_model.init(jax.random.PRNGKey(42), gojax.new_states(batch_size=2, board_size=3))
        self.assertIsInstance(params, dict)
        self.assertEqual(len(params), 0)

    @flagsaver.flagsaver(board_size=3, embed_model='identity', value_model='tromp_taylor',
                         policy_model='tromp_taylor', transition_model='real')
    def test_make_model_tromp_taylor_model_runs(self):
        go_model = hk.without_apply_rng(models.make_model(FLAGS.board_size))
        new_states = gojax.new_states(batch_size=1, board_size=3)
        params = go_model.init(jax.random.PRNGKey(42), new_states)

        embed_model = go_model.apply[models.EMBED_INDEX]
        value_model = go_model.apply[models.VALUE_INDEX]
        policy_model = go_model.apply[models.POLICY_INDEX]
        transition_model = go_model.apply[models.TRANSITION_INDEX]

        embeds = embed_model(params, new_states)
        np.testing.assert_array_equal(value_model(params, embeds), [0])
        np.testing.assert_array_equal(policy_model(params, embeds),
                                      [[9, 9, 9, 9, 9, 9, 9, 9, 9, 0]])
        all_transitions = transition_model(params, embeds)
        chex.assert_shape(all_transitions, (1, 10, 6, 3, 3))
        np.testing.assert_array_equal(value_model(params, all_transitions[:, 0]), [-9])
        np.testing.assert_array_equal(policy_model(params, all_transitions[:, 0]),
                                      [[-9, 0, 0, 0, 0, 0, 0, 0, 0, -9]])


if __name__ == '__main__':
    absltest.main()
