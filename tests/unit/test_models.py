"""Tests model.py."""
# pylint: disable=missing-function-docstring,duplicate-code,unnecessary-lambda
import os
import tempfile

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, flagsaver, parameterized

import gojax
from muzero_gojax import main, models

FLAGS = main.FLAGS


class ModelsTestCase(chex.TestCase):
    """Tests models.py."""

    def setUp(self):
        FLAGS.mark_as_parsed()

    def test_load_model_preserves_build_config(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            with flagsaver.flagsaver(
                    save_dir=tmpdirname,
                    embed_model='NonSpatialConvEmbed',
                    value_model='Linear3DValue',
                    policy_model='Linear3DPolicy',
                    transition_model='NonSpatialConvTransition'):
                rng_key = jax.random.PRNGKey(FLAGS.rng)
                all_models_build_config = models.get_all_models_build_config(
                    FLAGS.board_size)
                model, params = models.build_model_with_params(
                    all_models_build_config, rng_key)
                go_state = jax.random.normal(
                    rng_key, (1024, 6, FLAGS.board_size, FLAGS.board_size))
                params = model.init(rng_key, go_state)
                models.save_model(params, all_models_build_config,
                                  FLAGS.save_dir)
                _, _, loaded_config = models.load_model(FLAGS.save_dir)
                self.assertEqual(loaded_config, all_models_build_config)

    def test_load_model_has_same_output_as_original(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            with flagsaver.flagsaver(
                    save_dir=tmpdirname,
                    embed_model='NonSpatialConvEmbed',
                    value_model='Linear3DValue',
                    policy_model='Linear3DPolicy',
                    transition_model='NonSpatialConvTransition'):
                rng_key = jax.random.PRNGKey(FLAGS.rng)
                all_models_build_config = models.get_all_models_build_config(
                    FLAGS.board_size)
                model, params = models.build_model_with_params(
                    all_models_build_config, rng_key)
                go_state = jax.random.bernoulli(
                    rng_key,
                    shape=(FLAGS.batch_size, gojax.NUM_CHANNELS,
                           FLAGS.board_size, FLAGS.board_size))
                params = model.init(rng_key, go_state)
                expected_output = model.apply[models.VALUE_INDEX](params,
                                                                  rng_key,
                                                                  go_state)
                models.save_model(params, all_models_build_config,
                                  FLAGS.save_dir)
                new_go_model, loaded_params, _ = models.load_model(
                    FLAGS.save_dir)
                np.testing.assert_allclose(
                    new_go_model.apply[models.VALUE_INDEX](
                        loaded_params, rng_key, go_state).astype(jnp.float32),
                    expected_output.astype(jnp.float32),
                    rtol=1)

    def test_get_benchmarks_loads_trained_models(self):
        all_models_build_config = models.get_all_models_build_config(
            FLAGS.board_size)
        _, params = models.build_model_with_params(
            all_models_build_config, jax.random.PRNGKey(FLAGS.rng))
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_dir = os.path.join(tmpdirname, 'new_model')
            models.save_model(params, all_models_build_config, model_dir)
            with flagsaver.flagsaver(trained_models_dir=tmpdirname):
                self.assertTrue(os.path.exists(FLAGS.trained_models_dir))
                benchmarks = models.get_benchmarks(FLAGS.board_size)
        self.assertEqual(benchmarks[-1].name, model_dir)

    @parameterized.named_parameters(
        dict(testcase_name=models.IdentityEmbed.__name__,
             model_class=models.IdentityEmbed,
             embed_dim=6,
             expected_shape=(2, 6, 3, 3)),
        dict(testcase_name=models.AmplifiedEmbed.__name__,
             model_class=models.AmplifiedEmbed,
             embed_dim=6,
             expected_shape=(2, 6, 3, 3)),
        dict(testcase_name=models.CanonicalEmbed.__name__,
             model_class=models.CanonicalEmbed,
             embed_dim=6,
             expected_shape=(2, 6, 3, 3)),
        dict(testcase_name=models.NonSpatialConvEmbed.__name__,
             model_class=models.NonSpatialConvEmbed,
             embed_dim=2,
             expected_shape=(2, 2, 3, 3)),
        dict(testcase_name=models.ResNetV2Embed.__name__,
             model_class=models.ResNetV2Embed,
             embed_dim=2,
             expected_shape=(2, 2, 3, 3)),
        dict(testcase_name=models.CanonicalResNetV2Embed.__name__,
             model_class=models.CanonicalResNetV2Embed,
             embed_dim=2,
             expected_shape=(2, 2, 3, 3)),
        dict(testcase_name=models.CanonicalResNetV3Embed.__name__,
             model_class=models.CanonicalResNetV3Embed,
             embed_dim=256,
             expected_shape=(2, 256, 3, 3))  # TODO: Update to embed_dim.
    )
    def test_embed_model_output_matches_expected_shape(self, model_class,
                                                       embed_dim,
                                                       expected_shape):
        with flagsaver.flagsaver(batch_size=2,
                                 board_size=3,
                                 hdim=4,
                                 embed_dim=embed_dim,
                                 embed_model=model_class.__name__):
            rng_key = jax.random.PRNGKey(FLAGS.rng)
            go_model, params = models.build_model_with_params(
                models.get_all_models_build_config(FLAGS.board_size), rng_key)
            states = gojax.new_states(FLAGS.board_size, FLAGS.batch_size)
            output = go_model.apply[models.EMBED_INDEX](params, rng_key,
                                                        states)
            chex.assert_shape(output, expected_shape)

    @parameterized.named_parameters(
        # TODO: Update to embed_dim.
        # Value
        dict(testcase_name=models.RandomValue.__name__,
             model_class=models.RandomValue,
             embed_dim=2,
             expected_shape=(2, 2, 3, 3)),
        dict(testcase_name=models.LinearConvValue.__name__,
             model_class=models.LinearConvValue,
             embed_dim=2,
             expected_shape=(2, 2, 3, 3)),
        dict(testcase_name=models.Linear3DValue.__name__,
             model_class=models.Linear3DValue,
             embed_dim=2,
             expected_shape=(2, 2, 3, 3)),
        dict(testcase_name=models.TrompTaylorValue.__name__,
             model_class=models.TrompTaylorValue,
             embed_dim=2,
             expected_shape=(2, 2, 3, 3)),
        dict(testcase_name=models.ResNetV2Value.__name__,
             model_class=models.ResNetV2Value,
             embed_dim=2,
             expected_shape=(2, 2, 3, 3)),
        dict(testcase_name=models.ResNetV3Value.__name__,
             model_class=models.ResNetV3Value,
             embed_dim=256,
             expected_shape=(2, 2, 3, 3)),  # TODO: Update to embed_dim.
        # Policy
        dict(testcase_name=models.RandomPolicy.__name__,
             model_class=models.RandomPolicy,
             embed_dim=2,
             expected_shape=(2, 10)),
        dict(testcase_name=models.LinearConvPolicy.__name__,
             model_class=models.LinearConvPolicy,
             embed_dim=2,
             expected_shape=(2, 10)),
        dict(testcase_name=models.SingleLayerConvPolicy.__name__,
             model_class=models.SingleLayerConvPolicy,
             embed_dim=2,
             expected_shape=(2, 10)),
        dict(testcase_name=models.Linear3DPolicy.__name__,
             model_class=models.Linear3DPolicy,
             embed_dim=2,
             expected_shape=(2, 10)),
        dict(testcase_name=models.NonSpatialConvPolicy.__name__,
             model_class=models.NonSpatialConvPolicy,
             embed_dim=2,
             expected_shape=(2, 10)),
        dict(testcase_name=models.TrompTaylorPolicy.__name__,
             model_class=models.TrompTaylorPolicy,
             embed_dim=2,
             expected_shape=(2, 10)),
        dict(testcase_name=models.ResNetV3Policy.__name__,
             model_class=models.ResNetV3Policy,
             embed_dim=256,
             expected_shape=(2, 10)),  # TODO: Update to embed_dim.
        # Transition
        dict(testcase_name=models.RealTransition.__name__,
             model_class=models.RealTransition,
             embed_dim=6,
             expected_shape=(2, 10, 6, 3, 3)),
        dict(testcase_name=models.BlackRealTransition.__name__,
             model_class=models.BlackRealTransition,
             embed_dim=6,
             expected_shape=(2, 10, 6, 3, 3)),
        dict(testcase_name=models.RandomTransition.__name__,
             model_class=models.RandomTransition,
             embed_dim=2,
             expected_shape=(2, 10, 2, 3, 3)),
        dict(testcase_name=models.NonSpatialConvTransition.__name__,
             model_class=models.NonSpatialConvTransition,
             embed_dim=2,
             expected_shape=(2, 10, 2, 3, 3)),
        dict(testcase_name=models.ResNetV2Transition.__name__,
             model_class=models.ResNetV2Transition,
             embed_dim=2,
             expected_shape=(2, 10, 2, 3, 3)),
        dict(testcase_name=models.ResNetV3Transition.__name__,
             model_class=models.ResNetV3Transition,
             embed_dim=256,
             expected_shape=(2, 10, 256, 3, 3)),  # TODO: Update to embed_dim.
    )
    def test_non_embed_model_matches_expected_shape(self, model_class,
                                                    embed_dim, expected_shape):
        with flagsaver.flagsaver(batch_size=2,
                                 board_size=3,
                                 hdim=4,
                                 embed_dim=embed_dim):
            model_config = models.ModelBuildConfig(board_size=FLAGS.board_size,
                                                   hdim=FLAGS.hdim,
                                                   embed_dim=FLAGS.embed_dim)
            submodel_config = models.SubModelBuildConfig()
            model = hk.transform(
                lambda x: model_class(model_config, submodel_config)(x))
            embeds = jnp.zeros((FLAGS.batch_size, FLAGS.embed_dim,
                                FLAGS.board_size, FLAGS.board_size))
            params = model.init(jax.random.PRNGKey(42), embeds)
            output = model.apply(params, jax.random.PRNGKey(42), embeds)
            chex.assert_shape(output, expected_shape)

    def test_embed_canonical_swaps_white_turns(self):
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
        with flagsaver.flagsaver(embed_model='CanonicalEmbed'):
            go_model, params = models.build_model_with_params(
                models.get_all_models_build_config(FLAGS.board_size),
                jax.random.PRNGKey(42))
        np.testing.assert_array_equal(
            go_model.apply[models.EMBED_INDEX](params, None, states),
            expected_embedding)

    @flagsaver.flagsaver(board_size=3,
                         embed_model='IdentityEmbed',
                         value_model='Linear3DValue',
                         policy_model='Linear3DPolicy',
                         transition_model='RealTransition')
    def test_real_transition_model_outputs_all_children_from_start_state(self):
        all_models_build_config = models.get_all_models_build_config(
            FLAGS.board_size)
        go_model, params = models.build_model_with_params(
            all_models_build_config, jax.random.PRNGKey(FLAGS.rng))
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

    @flagsaver.flagsaver(transition_model='BlackRealTransition')
    def test_black_real_transition_outputs_all_children_from_empty_passed_state(
            self):
        all_models_build_config = models.get_all_models_build_config(
            FLAGS.board_size)
        go_model, params = models.build_model_with_params(
            all_models_build_config, jax.random.PRNGKey(FLAGS.rng))
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

    def test_tromp_taylor_value_model_outputs_amplified_canonical_area_differences(
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
        tromp_taylor_value = hk.without_apply_rng(
            hk.transform(lambda x: models.TrompTaylorValue(
                model_config=models.ModelBuildConfig(
                    board_size=3, hdim=4, embed_dim=6),
                submodel_config=models.SubModelBuildConfig())(x)))
        params = tromp_taylor_value.init(None, states)
        self.assertEmpty(params)
        np.testing.assert_array_equal(tromp_taylor_value.apply(params, states),
                                      (gojax.compute_areas(
                                          gojax.decode_states("""
                                    _ B B
                                    _ W _
                                    _ _ _
                                    TURN=B

                                    _ B _
                                    _ _ _
                                    _ _ _
                                    TURN=B
                                      """)).astype('float32') * 2 - 1) * 100)

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
            hk.transform(lambda x: models.TrompTaylorPolicy(
                model_config=models.ModelBuildConfig(
                    board_size=3, hdim=4, embed_dim=6),
                submodel_config=models.SubModelBuildConfig())(x)))
        params = tromp_taylor_policy.init(None, states)
        self.assertEmpty(params)
        np.testing.assert_array_equal(
            tromp_taylor_policy.apply(params, states),
            [[2, 1, 1, 3, 1, 2, 2, 2, 2, 1], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]])

    @flagsaver.flagsaver(board_size=3,
                         embed_model='IdentityEmbed',
                         area_model='RandomArea',
                         value_model='RandomValue',
                         policy_model='RandomPolicy',
                         transition_model='RandomTransition')
    def test_make_random_model_has_empty_params(self):
        all_models_build_config = models.get_all_models_build_config(
            FLAGS.board_size)
        go_model, params = models.build_model_with_params(
            all_models_build_config, jax.random.PRNGKey(FLAGS.rng))
        self.assertIsInstance(go_model, hk.MultiTransformed)
        self.assertIsInstance(params, dict)
        self.assertEqual(len(params), 0)


if __name__ == '__main__':
    absltest.main()
    absltest.main()
