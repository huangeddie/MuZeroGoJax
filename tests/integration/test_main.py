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

    @flagsaver.flagsaver(board_size=5,
                         trajectory_length=24,
                         batch_size=16,
                         training_steps=2,
                         optimizer='adamw',
                         learning_rate=1,
                         embed_model='identity',
                         transition_model='real',
                         nlayers=0,
                         value_model='non_spatial_conv',
                         policy_model='non_spatial_conv',
                         temperature=0.02)
    def test_real_linear_model_learns_to_avoid_occupied_spaces(self):
        go_model, init_params = models.build_model(FLAGS.board_size,
                                                   FLAGS.dtype)
        states = gojax.decode_states("""
                                    _ B _ W _
                                    W _ B _ W
                                    _ W _ B _
                                    B _ W _ B
                                    _ B _ W _
                                    """)

        trained_params, _ = train.train_model(go_model, init_params,
                                              FLAGS.board_size, FLAGS.dtype)

        rng_key = jax.random.PRNGKey(FLAGS.rng)
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

    def test_real_linear_model_caps_at_75_percent_value_acc(self):
        with flagsaver.flagsaver(board_size=5,
                                 trajectory_length=24,
                                 batch_size=128,
                                 training_steps=50,
                                 eval_frequency=1,
                                 optimizer='adamw',
                                 learning_rate=1e-1,
                                 embed_model='identity',
                                 transition_model='real',
                                 value_model='non_spatial_conv',
                                 policy_model='non_spatial_conv',
                                 self_play_model='random',
                                 nlayers=0):
            go_model, init_params = models.build_model(FLAGS.board_size,
                                                       FLAGS.dtype)

            linear_train_metrics: pd.DataFrame
            _, linear_train_metrics = train.train_model(
                go_model, init_params, FLAGS.board_size, FLAGS.dtype)

        self.assertBetween(linear_train_metrics.iloc[-1]['value.acc'], 0.45,
                           0.55)
        self.assertBetween(linear_train_metrics.iloc[-1]['value.loss'], 0.52,
                           0.58)

    def test_real_two_layer_model_caps_at_80_percent_value_acc(self):
        with flagsaver.flagsaver(board_size=5,
                                 trajectory_length=24,
                                 batch_size=128,
                                 training_steps=20,
                                 eval_frequency=1,
                                 optimizer='adamw',
                                 learning_rate=1e-2,
                                 embed_model='identity',
                                 transition_model='real',
                                 value_model='non_spatial_conv',
                                 policy_model='non_spatial_conv',
                                 self_play_model='random',
                                 nlayers=1,
                                 hdim=256):
            go_model, init_params = models.build_model(FLAGS.board_size,
                                                       FLAGS.dtype)

            mlp_train_metrics: pd.DataFrame
            _, mlp_train_metrics = train.train_model(go_model, init_params,
                                                     FLAGS.board_size,
                                                     FLAGS.dtype)

        self.assertBetween(mlp_train_metrics.iloc[-1]['value.acc'], 0.45, 0.55)
        self.assertBetween(mlp_train_metrics.iloc[-1]['value.loss'], 0.5, 0.6)


if __name__ == '__main__':
    unittest.main()
