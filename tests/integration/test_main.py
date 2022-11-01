"""Integration (main) tests."""
#pylint: disable=missing-class-docstring,missing-function-docstring

import unittest

import chex
import gojax
import jax
import jax.numpy as jnp
import numpy as np
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
                         value_model='linear_conv',
                         policy_model='linear_conv',
                         temperature=0.02)
    def test_real_linear_model_learns_to_avoid_occupied_spaces(self):
        go_model, init_params = models.build_model(FLAGS.board_size)
        states = gojax.decode_states("""
                                    _ B _ W _
                                    W _ B _ W
                                    _ W _ B _
                                    B _ W _ B
                                    _ B _ W _
                                    """)

        trained_params, _ = train.train_model(go_model, init_params,
                                              FLAGS.board_size)

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


if __name__ == '__main__':
    unittest.main()
