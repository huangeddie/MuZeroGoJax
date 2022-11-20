"""Integration (main) tests."""
#pylint: disable=missing-class-docstring,missing-function-docstring

import unittest

import chex
import gojax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from absl.testing import flagsaver

from muzero_gojax import main, models, train

FLAGS = main.FLAGS


class MainTestCase(chex.TestCase):

    def setUp(self):
        FLAGS.mark_as_parsed()

    @flagsaver.flagsaver(batch_size=8,
                         training_steps=2,
                         optimizer='adamw',
                         learning_rate=1,
                         embed_model='identity',
                         transition_model='real',
                         value_model='tromp_taylor',
                         nlayers=0,
                         policy_model='non_spatial_conv',
                         temperature=0.1,
                         dtype='float32',
                         sample_action_size=26,
                         self_play_model='random')
    def test_real_linear_policy_learns_to_avoid_occupied_spaces(self):
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        go_model, init_params = models.build_model_with_params(
            FLAGS.board_size, FLAGS.dtype, rng_key)
        states = gojax.decode_states("""
                                    _ B _ W _
                                    W _ B _ W
                                    _ W _ B _
                                    B _ W _ B
                                    _ B _ W _
                                    """)

        trained_params, _ = train.train_model(go_model, init_params,
                                              FLAGS.board_size, FLAGS.dtype,
                                              rng_key)

        embeddings = go_model.apply[models.EMBED_INDEX](trained_params,
                                                        rng_key, states)
        policy_logits = go_model.apply[models.POLICY_INDEX](
            trained_params, rng_key, embeddings).astype('float32')
        policy = jnp.squeeze(jax.nn.softmax(policy_logits, axis=-1), axis=0)
        action_probs = policy[:-1]
        pass_prob = policy[-1]

        # 1 / 26 ~ 0.038
        np.testing.assert_array_less(
            action_probs[1::2],
            jnp.full_like(action_probs[1::2], fill_value=0.038))
        np.testing.assert_array_less(
            jnp.full_like(action_probs[0::2], fill_value=0.038),
            action_probs[0::2])
        self.assertLessEqual(pass_prob, 0.038)

    @flagsaver.flagsaver(batch_size=128,
                         training_steps=50,
                         eval_frequency=1,
                         optimizer='adamw',
                         learning_rate=1e-1,
                         embed_model='identity',
                         transition_model='real',
                         value_model='non_spatial_conv',
                         self_play_model='random',
                         nlayers=0)
    def test_real_linear_caps_at_55_percent_value_acc(self):
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        go_model, init_params = models.build_model_with_params(
            FLAGS.board_size, FLAGS.dtype, rng_key)

        linear_train_metrics: pd.DataFrame
        _, linear_train_metrics = train.train_model(go_model, init_params,
                                                    FLAGS.board_size,
                                                    FLAGS.dtype, rng_key)

        self.assertAlmostEqual(linear_train_metrics.iloc[-1]['value_acc'],
                               0.55,
                               delta=0.05)

    @flagsaver.flagsaver(batch_size=128,
                         training_steps=20,
                         eval_frequency=1,
                         optimizer='adamw',
                         learning_rate=1e-2,
                         embed_model='identity',
                         transition_model='real',
                         value_model='non_spatial_quad_conv',
                         self_play_model='random',
                         nlayers=0)
    def test_real_quad_caps_at_55_percent_value_acc(self):
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        go_model, init_params = models.build_model_with_params(
            FLAGS.board_size, FLAGS.dtype, rng_key)

        linear_train_metrics: pd.DataFrame
        _, linear_train_metrics = train.train_model(go_model, init_params,
                                                    FLAGS.board_size,
                                                    FLAGS.dtype, rng_key)

        self.assertAlmostEqual(linear_train_metrics.iloc[-1]['value_acc'],
                               0.55,
                               delta=0.05)

    @flagsaver.flagsaver(batch_size=256,
                         training_steps=20,
                         eval_frequency=1,
                         optimizer='adamw',
                         learning_rate=1e-3,
                         embed_model='identity',
                         transition_model='real',
                         policy_model='random',
                         value_model='non_spatial_conv',
                         self_play_model='random',
                         nlayers=1,
                         hdim=256)
    def test_real_mlp_caps_at_50_percent_value_acc(self):
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        go_model, init_params = models.build_model_with_params(
            FLAGS.board_size, FLAGS.dtype, rng_key)

        linear_train_metrics: pd.DataFrame
        _, linear_train_metrics = train.train_model(go_model, init_params,
                                                    FLAGS.board_size,
                                                    FLAGS.dtype, rng_key)

        self.assertAlmostEqual(linear_train_metrics.iloc[-1]['value_acc'],
                               0.50,
                               delta=0.05)

    @flagsaver.flagsaver(batch_size=128,
                         training_steps=1,
                         eval_frequency=1,
                         optimizer='adamw',
                         learning_rate=1e-2,
                         embed_model='identity',
                         transition_model='real',
                         value_model='tromp_taylor',
                         self_play_model='random')
    def test_tromp_taylor_caps_at_75_percent_value_acc(self):
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        go_model, init_params = models.build_model_with_params(
            FLAGS.board_size, FLAGS.dtype, rng_key)

        mlp_train_metrics: pd.DataFrame
        _, mlp_train_metrics = train.train_model(go_model, init_params,
                                                 FLAGS.board_size, FLAGS.dtype,
                                                 rng_key)

        self.assertAlmostEqual(mlp_train_metrics.iloc[-1]['value_acc'],
                               0.75,
                               delta=0.05)

    @flagsaver.flagsaver(batch_size=128,
                         training_steps=1,
                         eval_frequency=1,
                         optimizer='adamw',
                         embed_model='identity',
                         transition_model='real',
                         value_model='piece_counter',
                         self_play_model='random')
    def test_piece_counter_caps_at_72_percent_value_acc(self):
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        go_model, init_params = models.build_model_with_params(
            FLAGS.board_size, FLAGS.dtype, rng_key)

        mlp_train_metrics: pd.DataFrame
        _, mlp_train_metrics = train.train_model(go_model, init_params,
                                                 FLAGS.board_size, FLAGS.dtype,
                                                 rng_key)

        self.assertAlmostEqual(mlp_train_metrics.iloc[-1]['value_acc'],
                               0.72,
                               delta=0.05)

    @flagsaver.flagsaver(batch_size=128,
                         training_steps=1,
                         eval_frequency=1,
                         optimizer='adamw',
                         embed_model='identity',
                         transition_model='real',
                         value_model='heuristic_quad_conv',
                         self_play_model='random')
    def test_heuristic_quad_value_caps_at_72_percent_acc(self):
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        go_model, init_params = models.build_model_with_params(
            FLAGS.board_size, FLAGS.dtype, rng_key)

        mlp_train_metrics: pd.DataFrame
        _, mlp_train_metrics = train.train_model(go_model, init_params,
                                                 FLAGS.board_size, FLAGS.dtype,
                                                 rng_key)

        self.assertAlmostEqual(mlp_train_metrics.iloc[-1]['value_acc'],
                               0.72,
                               delta=0.05)


if __name__ == '__main__':
    unittest.main()
