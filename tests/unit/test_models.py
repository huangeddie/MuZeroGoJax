"""Tests model.py."""
# pylint: disable=missing-function-docstring,duplicate-code,unnecessary-lambda
import os
import tempfile

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
from muzero_gojax import train
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

    def test_save_model_saves_model_with_bfloat16_type(self):
        """Saving bfloat16 model weights should be ok."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            with flagsaver.flagsaver(save_dir=tmpdirname,
                                     embed_model='non_spatial_conv',
                                     value_model='linear',
                                     policy_model='linear',
                                     transition_model='non_spatial_conv'):
                params = {'foo': jnp.array(0, dtype='bfloat16')}
                model_dir = os.path.join(tmpdirname,
                                         train.hash_model_flags(FLAGS))
                models.save_model(params, model_dir)
                self.assertTrue(os.path.exists(model_dir))

    def test_load_model_bfloat16(self):
        """Loading bfloat16 model weights should be ok."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            with flagsaver.flagsaver(save_dir=tmpdirname,
                                     embed_model='non_spatial_conv',
                                     value_model='linear',
                                     policy_model='linear',
                                     transition_model='non_spatial_conv'):
                model = hk.transform(
                    lambda x: models.value.Linear3DValue(FLAGS)(x))
                rng_key = jax.random.PRNGKey(FLAGS.rng)
                go_state = jax.random.normal(rng_key, (1024, 6, 19, 19))
                params = model.init(rng_key, go_state)
                params = jax.tree_util.tree_map(lambda x: x.astype('bfloat16'),
                                                params)
                expected_output = model.apply(params, rng_key, go_state)
                model_dir = os.path.join(tmpdirname,
                                         train.hash_model_flags(FLAGS))
                models.save_model(params, model_dir)
                params = models.load_tree_array(
                    os.path.join(model_dir, 'params.npz'), 'bfloat16')
                np.testing.assert_array_equal(
                    model.apply(params, rng_key, go_state), expected_output)

    def test_load_model_float32(self):
        """Loading float32 model weights should be ok."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            with flagsaver.flagsaver(save_dir=tmpdirname,
                                     embed_model='non_spatial_conv',
                                     value_model='linear',
                                     policy_model='linear',
                                     transition_model='non_spatial_conv'):
                model = hk.transform(
                    lambda x: models.value.Linear3DValue(FLAGS)(x))
                rng_key = jax.random.PRNGKey(FLAGS.rng)
                go_state = jax.random.normal(rng_key, (1024, 6, 19, 19))
                params = model.init(rng_key, go_state)
                expected_output = model.apply(params, rng_key, go_state)
                model_dir = os.path.join(tmpdirname,
                                         train.hash_model_flags(FLAGS))
                models.save_model(params, model_dir)
                params = models.load_tree_array(
                    os.path.join(model_dir, 'params.npz'), 'float32')
                np.testing.assert_allclose(model.apply(params, rng_key,
                                                       go_state),
                                           expected_output.astype('float32'),
                                           rtol=0.1)

    def test_load_model_bfloat16_to_float32(self):
        """Loading float32 model weights from saved bfloat16 weights should be ok."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            with flagsaver.flagsaver(save_dir=tmpdirname,
                                     embed_model='non_spatial_conv',
                                     value_model='linear',
                                     policy_model='linear',
                                     transition_model='non_spatial_conv'):
                model = hk.transform(
                    lambda x: models.value.Linear3DValue(FLAGS)(x))
                rng_key = jax.random.PRNGKey(FLAGS.rng)
                go_state = jax.random.normal(rng_key, (1024, 6, 19, 19))
                params = model.init(rng_key, go_state)
                params = jax.tree_util.tree_map(lambda x: x.astype('bfloat16'),
                                                params)
                expected_output = model.apply(params, rng_key, go_state)
                model_dir = os.path.join(tmpdirname,
                                         train.hash_model_flags(FLAGS))
                models.save_model(params, model_dir)
                params = models.load_tree_array(
                    os.path.join(model_dir, 'params.npz'), 'float32')
                np.testing.assert_allclose(model.apply(params, rng_key,
                                                       go_state),
                                           expected_output.astype('float32'),
                                           rtol=0.1)

    def test_load_model_float32_to_bfloat16_approximation(self):
        """Loading float32 model weights from bfloat16 should be ok with some inconsistencies."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            with flagsaver.flagsaver(save_dir=tmpdirname,
                                     embed_model='non_spatial_conv',
                                     value_model='linear',
                                     policy_model='linear',
                                     transition_model='non_spatial_conv',
                                     dtype='bfloat16'):
                model = hk.transform(
                    lambda x: models.value.Linear3DValue(FLAGS)(x))
                rng_key = jax.random.PRNGKey(FLAGS.rng)
                go_state = jax.random.normal(rng_key, (1024, 6, 19, 19))
                params = model.init(rng_key, go_state)
                expected_output = model.apply(params, rng_key, go_state)
                model_dir = os.path.join(tmpdirname,
                                         train.hash_model_flags(FLAGS))
                models.save_model(params, model_dir)
                params = models.load_tree_array(
                    os.path.join(model_dir, 'params.npz'), FLAGS.dtype)
                np.testing.assert_allclose(model.apply(params, rng_key,
                                                       go_state),
                                           expected_output.astype('float32'),
                                           rtol=1)

    @parameterized.named_parameters(
        dict(testcase_name=embed.IdentityEmbed.__name__,
             model_class=embed.IdentityEmbed,
             embed_dim=6,
             expected_shape=(2, 6, 3, 3)),
        dict(testcase_name=embed.AmplifiedEmbed.__name__,
             model_class=embed.AmplifiedEmbed,
             embed_dim=6,
             expected_shape=(2, 6, 3, 3)),
        dict(testcase_name=embed.BlackPerspectiveEmbed.__name__,
             model_class=embed.BlackPerspectiveEmbed,
             embed_dim=6,
             expected_shape=(2, 6, 3, 3)),
        dict(testcase_name=embed.NonSpatialConvEmbed.__name__,
             model_class=embed.NonSpatialConvEmbed,
             embed_dim=2,
             expected_shape=(2, 2, 3, 3)),
        dict(testcase_name=embed.ResNetV2Embed.__name__,
             model_class=embed.ResNetV2Embed,
             embed_dim=2,
             expected_shape=(2, 2, 3, 3)))
    def test_embed_model_output_type_and_shape(self, model_class, embed_dim,
                                               expected_shape):
        with flagsaver.flagsaver(batch_size=2,
                                 board_size=3,
                                 hdim=4,
                                 embed_dim=embed_dim):
            model = hk.transform(lambda x: model_class(FLAGS)(x))
            states = gojax.new_states(FLAGS.board_size, FLAGS.batch_size)
            params = model.init(jax.random.PRNGKey(42), states)
            output = model.apply(params, jax.random.PRNGKey(42), states)
            chex.assert_shape(output, expected_shape)
            chex.assert_type(output, 'bfloat16')

    @parameterized.named_parameters(
        # Decode
        dict(testcase_name=decode.AmplifiedDecode.__name__,
             model_class=decode.AmplifiedDecode,
             embed_dim=2,
             expected_shape=(2, 6, 3, 3)),
        dict(testcase_name=decode.NonSpatialConvDecode.__name__,
             model_class=decode.NonSpatialConvDecode,
             embed_dim=2,
             expected_shape=(2, 6, 3, 3)),
        dict(testcase_name=decode.ResNetV2Decode.__name__,
             model_class=decode.ResNetV2Decode,
             embed_dim=2,
             expected_shape=(2, 6, 3, 3)),
        # Value
        dict(testcase_name=value.RandomValue.__name__,
             model_class=value.RandomValue,
             embed_dim=2,
             expected_shape=(2, )),
        dict(testcase_name=value.NonSpatialConvValue.__name__,
             model_class=value.NonSpatialConvValue,
             embed_dim=2,
             expected_shape=(2, )),
        dict(testcase_name=value.NonSpatialQuadConvValue.__name__,
             model_class=value.NonSpatialQuadConvValue,
             embed_dim=2,
             expected_shape=(2, )),
        dict(testcase_name=value.HeuristicQuadConvValue.__name__,
             model_class=value.HeuristicQuadConvValue,
             embed_dim=6,
             expected_shape=(2, )),
        dict(testcase_name=value.Linear3DValue.__name__,
             model_class=value.Linear3DValue,
             embed_dim=2,
             expected_shape=(2, )),
        dict(testcase_name=value.TrompTaylorValue.__name__,
             model_class=value.TrompTaylorValue,
             embed_dim=2,
             expected_shape=(2, )),
        dict(testcase_name=value.PieceCounterValue.__name__,
             model_class=value.PieceCounterValue,
             embed_dim=2,
             expected_shape=(2, )),
        dict(testcase_name=value.ResNetV2Value.__name__,
             model_class=value.ResNetV2Value,
             embed_dim=2,
             expected_shape=(2, )),
        # Policy
        dict(testcase_name=policy.RandomPolicy.__name__,
             model_class=policy.RandomPolicy,
             embed_dim=2,
             expected_shape=(2, 10)),
        dict(testcase_name=policy.Linear3DPolicy.__name__,
             model_class=policy.Linear3DPolicy,
             embed_dim=2,
             expected_shape=(2, 10)),
        dict(testcase_name=policy.NonSpatialConvPolicy.__name__,
             model_class=policy.NonSpatialConvPolicy,
             embed_dim=2,
             expected_shape=(2, 10)),
        dict(testcase_name=policy.TrompTaylorPolicy.__name__,
             model_class=policy.TrompTaylorPolicy,
             embed_dim=2,
             expected_shape=(2, 10)),
        # Transition
        dict(testcase_name=transition.RealTransition.__name__,
             model_class=transition.RealTransition,
             embed_dim=6,
             expected_shape=(2, 10, 6, 3, 3)),
        dict(testcase_name=transition.BlackRealTransition.__name__,
             model_class=transition.BlackRealTransition,
             embed_dim=6,
             expected_shape=(2, 10, 6, 3, 3)),
        dict(testcase_name=transition.RandomTransition.__name__,
             model_class=transition.RandomTransition,
             embed_dim=2,
             expected_shape=(2, 10, 2, 3, 3)),
        dict(testcase_name=transition.NonSpatialConvTransition.__name__,
             model_class=transition.NonSpatialConvTransition,
             embed_dim=2,
             expected_shape=(2, 10, 2, 3, 3)),
        dict(testcase_name=transition.ResNetV2ActionTransition.__name__,
             model_class=transition.ResNetV2ActionTransition,
             embed_dim=2,
             expected_shape=(2, 10, 2, 3, 3)),
    )
    def test_non_embed_model_output_type_and_shape(self, model_class,
                                                   embed_dim, expected_shape):
        with flagsaver.flagsaver(batch_size=2,
                                 board_size=3,
                                 hdim=4,
                                 embed_dim=embed_dim):
            model = hk.transform(lambda x: model_class(FLAGS)(x))
            embeds = jnp.zeros((FLAGS.batch_size, FLAGS.embed_dim,
                                FLAGS.board_size, FLAGS.board_size),
                               dtype='bfloat16')
            params = model.init(jax.random.PRNGKey(42), embeds)
            output = model.apply(params, jax.random.PRNGKey(42), embeds)
            chex.assert_shape(output, expected_shape)
            chex.assert_type(output, 'bfloat16')

    def test_embed_black_perspective_swaps_white_turns(self):
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
            hk.transform(lambda x: embed.BlackPerspectiveEmbed(
                model_params=base.ModelBuildParams(
                    board_size=3, hdim=4, embed_dim=6, nlayers=1))(x)))
        rng = jax.random.PRNGKey(42)
        params = embed_model.init(rng, states)
        self.assertEmpty(params)
        np.testing.assert_array_equal(embed_model.apply(params, states),
                                      expected_embedding)

    @flagsaver.flagsaver(board_size=3,
                         embed_model='identity',
                         value_model='linear',
                         policy_model='linear',
                         transition_model='real')
    def test_real_transition_model_outputs_all_children_from_start_state(self):
        go_model, params = models.build_model_with_params(
            FLAGS.board_size, FLAGS.dtype, jax.random.PRNGKey(FLAGS.rng))
        new_states = gojax.new_states(batch_size=1, board_size=3)

        transition_model = go_model.apply[models.TRANSITION_INDEX]
        transition_output = transition_model(params, None, new_states)
        expected_transition = jnp.expand_dims(gojax.decode_states(
            """
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
                              """,
            turn=gojax.WHITES_TURN),
                                              axis=0)
        np.testing.assert_array_equal(transition_output, expected_transition)

    @flagsaver.flagsaver(transition_model='black_perspective')
    def test_transition_black_perspective_outputs_all_children_from_empty_passed_state(
            self):
        go_model, params = models.build_model_with_params(
            FLAGS.board_size, FLAGS.dtype, jax.random.PRNGKey(FLAGS.rng))
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
                              """),
                                              axis=0)
        np.testing.assert_array_equal(transition_output, expected_transition)

    def test_tromp_taylor_value_model_outputs_area_differences(self):
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
            hk.transform(lambda x: models.value.TrompTaylorValue(
                model_params=base.ModelBuildParams(
                    board_size=3, hdim=4, embed_dim=6, nlayers=1))(x)))
        params = tromp_taylor_value.init(None, states)
        self.assertEmpty(params)
        np.testing.assert_array_equal(tromp_taylor_value.apply(params, states),
                                      [1, 9])

    def test_tromp_taylor_policy_model_outputs_next_state_area_differences(
            self):
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
            hk.transform(lambda x: models.policy.TrompTaylorPolicy(
                model_params=base.ModelBuildParams(
                    board_size=3, hdim=4, embed_dim=6, nlayers=1))(x)))
        params = tromp_taylor_policy.init(None, states)
        self.assertEmpty(params)
        np.testing.assert_array_equal(
            tromp_taylor_policy.apply(params, states),
            [[2, 1, 1, 3, 1, 2, 2, 2, 2, 1], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]])

    @flagsaver.flagsaver(board_size=3,
                         embed_model='identity',
                         decode_model='amplified',
                         value_model='random',
                         policy_model='random',
                         transition_model='random')
    def test_make_random_model_has_empty_params(self):
        go_model, params = models.build_model_with_params(
            FLAGS.board_size, FLAGS.dtype, jax.random.PRNGKey(FLAGS.rng))
        self.assertIsInstance(go_model, hk.MultiTransformed)
        self.assertIsInstance(params, dict)
        self.assertEqual(len(params), 0)

    @flagsaver.flagsaver(board_size=3,
                         embed_model='identity',
                         value_model='tromp_taylor',
                         policy_model='tromp_taylor',
                         transition_model='real')
    def test_tromp_taylor_model_outputs_expected_values_on_start_state(self):
        go_model, params = models.build_model_with_params(
            FLAGS.board_size, FLAGS.dtype, jax.random.PRNGKey(FLAGS.rng))
        new_states = gojax.new_states(batch_size=1, board_size=3)
        embed_model = go_model.apply[models.EMBED_INDEX]
        value_model = go_model.apply[models.VALUE_INDEX]
        policy_model = go_model.apply[models.POLICY_INDEX]
        transition_model = go_model.apply[models.TRANSITION_INDEX]
        embeds = embed_model(params, None, new_states)
        all_transitions = transition_model(params, None, embeds)

        np.testing.assert_array_equal(value_model(params, None, embeds), [0])
        np.testing.assert_array_equal(policy_model(params, None, embeds),
                                      [[9, 9, 9, 9, 9, 9, 9, 9, 9, 0]])
        chex.assert_shape(all_transitions, (1, 10, 6, 3, 3))
        np.testing.assert_array_equal(
            value_model(params, None, all_transitions[:, 0]), [-9])
        np.testing.assert_array_equal(
            policy_model(params, None, all_transitions[:, 0]),
            [[-9, 0, 0, 0, 0, 0, 0, 0, 0, -9]])


if __name__ == '__main__':
    absltest.main()
