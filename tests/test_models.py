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
from muzero_gojax.models import base
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
        dict(testcase_name=embed.Identity.__name__, model_class=embed.Identity, embed_dim=6,
             expected_shape=(2, 6, 3, 3)),
        dict(testcase_name=embed.BlackPerspective.__name__, model_class=embed.BlackPerspective,
             embed_dim=6, expected_shape=(2, 6, 3, 3)),
        dict(testcase_name=embed.BlackCnnLite.__name__, model_class=embed.BlackCnnLite, embed_dim=6,
             expected_shape=(2, 6, 3, 3)),
        dict(testcase_name=embed.LinearConvEmbed.__name__, model_class=embed.LinearConvEmbed,
             embed_dim=2, expected_shape=(2, 2, 3, 3)),
        dict(testcase_name=embed.CnnLiteEmbed.__name__, model_class=embed.CnnLiteEmbed, embed_dim=2,
             expected_shape=(2, 2, 3, 3)),
        dict(testcase_name=embed.ResNetV2Embed.__name__, model_class=embed.ResNetV2Embed,
             embed_dim=2, expected_shape=(2, 2, 3, 3)),  # Decode
        dict(testcase_name=decode.AmplifiedDecode.__name__, model_class=decode.AmplifiedDecode,
             embed_dim=2, expected_shape=(2, 6, 3, 3)),
        dict(testcase_name=decode.LinearConvDecode.__name__, model_class=decode.LinearConvDecode,
             embed_dim=2, expected_shape=(2, 6, 3, 3)),
        dict(testcase_name=decode.ResNetV2Decode.__name__, model_class=decode.ResNetV2Decode,
             embed_dim=2, expected_shape=(2, 6, 3, 3)),  # Value
        dict(testcase_name=value.RandomValue.__name__, model_class=value.RandomValue, embed_dim=2,
             expected_shape=(2,)),
        dict(testcase_name=value.LinearConvValue.__name__, model_class=value.LinearConvValue,
             embed_dim=2, expected_shape=(2,)),
        dict(testcase_name=value.Linear3DValue.__name__, model_class=value.Linear3DValue,
             embed_dim=2, expected_shape=(2,)),
        dict(testcase_name=value.CnnLiteValue.__name__, model_class=value.CnnLiteValue, embed_dim=2,
             expected_shape=(2,)),
        dict(testcase_name=value.ResnetMediumValue.__name__, model_class=value.ResnetMediumValue,
             embed_dim=2, expected_shape=(2,)),
        dict(testcase_name=value.TrompTaylorValue.__name__, model_class=value.TrompTaylorValue,
             embed_dim=2, expected_shape=(2,)),  # Policy
        dict(testcase_name=policy.RandomPolicy.__name__, model_class=policy.RandomPolicy,
             embed_dim=2, expected_shape=(2, 10)),
        dict(testcase_name=policy.Linear3DPolicy.__name__, model_class=policy.Linear3DPolicy,
             embed_dim=2, expected_shape=(2, 10)),
        dict(testcase_name=policy.LinearConvPolicy.__name__, model_class=policy.LinearConvPolicy,
             embed_dim=2, expected_shape=(2, 10)),
        dict(testcase_name=policy.CnnLitePolicy.__name__, model_class=policy.CnnLitePolicy,
             embed_dim=2, expected_shape=(2, 10)),
        dict(testcase_name=policy.ResnetMediumPolicy.__name__,
             model_class=policy.ResnetMediumPolicy, embed_dim=2, expected_shape=(2, 10)),
        dict(testcase_name=policy.TrompTaylorPolicy.__name__, model_class=policy.TrompTaylorPolicy,
             embed_dim=2, expected_shape=(2, 10)),  # Transition
        dict(testcase_name=transition.RealTransition.__name__,
             model_class=transition.RealTransition, embed_dim=6, expected_shape=(2, 10, 6, 3, 3)),
        dict(testcase_name=transition.BlackRealTransition.__name__,
             model_class=transition.BlackRealTransition, embed_dim=6,
             expected_shape=(2, 10, 6, 3, 3)),
        dict(testcase_name=transition.RandomTransition.__name__,
             model_class=transition.RandomTransition, embed_dim=2, expected_shape=(2, 10, 2, 3, 3)),
        dict(testcase_name=transition.LinearConvTransition.__name__,
             model_class=transition.LinearConvTransition, embed_dim=2,
             expected_shape=(2, 10, 2, 3, 3)),
        dict(testcase_name=transition.CnnLiteTransition.__name__,
             model_class=transition.CnnLiteTransition, embed_dim=2,
             expected_shape=(2, 10, 2, 3, 3)),
        dict(testcase_name=transition.ResnetMediumTransition.__name__,
             model_class=transition.ResnetMediumTransition, embed_dim=2,
             expected_shape=(2, 10, 2, 3, 3)),
        dict(testcase_name=transition.ResNetV2Transition.__name__,
             model_class=transition.ResNetV2Transition, embed_dim=2,
             expected_shape=(2, 10, 2, 3, 3)),
        dict(testcase_name=transition.ResNetV2ActionEmbedTransition.__name__,
             model_class=transition.ResNetV2ActionEmbedTransition, embed_dim=2,
             expected_shape=(2, 10, 2, 3, 3)), )
    def test_model_output(self, model_class, embed_dim, expected_shape):
        with flagsaver.flagsaver(board_size=3, hdim=4, embed_dim=embed_dim):
            model = hk.transform(lambda x: model_class(FLAGS)(x))
            states = gojax.new_states(batch_size=2, board_size=3)
            params = model.init(jax.random.PRNGKey(42), states)
            output = model.apply(params, jax.random.PRNGKey(42), states)
            chex.assert_shape(output, expected_shape)
            chex.assert_type(output, 'bfloat16')

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
        embed_model = hk.without_apply_rng(hk.transform(lambda x: embed.BlackPerspective(
            model_params=base.ModelParams(board_size=3, hdim=4, embed_dim=6, nlayers=1))(x)))
        rng = jax.random.PRNGKey(42)
        params = embed_model.init(rng, states)
        self.assertEmpty(params)
        np.testing.assert_array_equal(embed_model.apply(params, states), expected_embedding)

    @flagsaver.flagsaver(board_size=3, embed_model='identity', value_model='linear',
                         policy_model='linear', transition_model='real')
    def test_get_real_transition_model_output(self):
        go_model, params = models.make_model(board_size=3)
        new_states = gojax.new_states(batch_size=1, board_size=3)

        transition_model = go_model.apply[models.TRANSITION_INDEX]
        transition_output = transition_model(params, None, new_states)
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

    @flagsaver.flagsaver(transition_model='black_perspective')
    def test_transition_black_perspective_output(self):
        go_model, params = models.make_model(board_size=3)
        new_states = gojax.new_states(batch_size=1, board_size=3)

        transition_model = go_model.apply[models.TRANSITION_INDEX]
        transition_output = transition_model(params, None, new_states)
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
        tromp_taylor_value = hk.without_apply_rng(hk.transform(
            lambda x: models.value.TrompTaylorValue(
                model_params=base.ModelParams(board_size=3, hdim=4, embed_dim=6, nlayers=1))(x)))
        params = tromp_taylor_value.init(None, states)
        self.assertEmpty(params)
        np.testing.assert_array_equal(tromp_taylor_value.apply(params, states), [1, 9])

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
        tromp_taylor_policy = hk.without_apply_rng(hk.transform(
            lambda x: models.policy.TrompTaylorPolicy(
                model_params=base.ModelParams(board_size=3, hdim=4, embed_dim=6, nlayers=1))(x)))
        params = tromp_taylor_policy.init(None, states)
        self.assertEmpty(params)
        np.testing.assert_array_equal(tromp_taylor_policy.apply(params, states),
                                      [[2, 1, 1, 3, 1, 2, 2, 2, 2, 1],
                                       [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]])

    @flagsaver.flagsaver(board_size=3, embed_model='identity', value_model='random',
                         policy_model='random', transition_model='random')
    def test_make_random_model_params(self):
        go_model, params = models.make_model(board_size=3)
        self.assertIsInstance(go_model, hk.MultiTransformed)
        self.assertIsInstance(params, dict)
        self.assertEqual(len(params), 0)

    @flagsaver.flagsaver(board_size=3, embed_model='identity', value_model='tromp_taylor',
                         policy_model='tromp_taylor', transition_model='real')
    def test_make_model_tromp_taylor_model_runs(self):
        go_model, params = models.make_model(board_size=3)
        new_states = gojax.new_states(batch_size=1, board_size=3)

        embed_model = go_model.apply[models.EMBED_INDEX]
        value_model = go_model.apply[models.VALUE_INDEX]
        policy_model = go_model.apply[models.POLICY_INDEX]
        transition_model = go_model.apply[models.TRANSITION_INDEX]

        embeds = embed_model(params, None, new_states)
        np.testing.assert_array_equal(value_model(params, None, embeds), [0])
        np.testing.assert_array_equal(policy_model(params, None, embeds),
                                      [[9, 9, 9, 9, 9, 9, 9, 9, 9, 0]])
        all_transitions = transition_model(params, None, embeds)
        chex.assert_shape(all_transitions, (1, 10, 6, 3, 3))
        np.testing.assert_array_equal(value_model(params, None, all_transitions[:, 0]), [-9])
        np.testing.assert_array_equal(policy_model(params, None, all_transitions[:, 0]),
                                      [[-9, 0, 0, 0, 0, 0, 0, 0, 0, -9]])


if __name__ == '__main__':
    absltest.main()
