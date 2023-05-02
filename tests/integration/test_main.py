"""Integration (main) tests."""
#pylint: disable=missing-class-docstring,missing-function-docstring

import os
import tempfile
import unittest

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import flagsaver

import gojax
from muzero_gojax import main, manager, models

FLAGS = main.FLAGS


class MainTestCase(chex.TestCase):

    def setUp(self):
        FLAGS.mark_as_parsed()

    @flagsaver.flagsaver(skip_play=True, skip_plot=True)
    def test_default_flags_runs_main_with_no_error(self):
        main.main(None)

    def test_saved_pmap_model_loads_correctly(self):
        os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
        self.assertEqual(jax.device_count(), 8)
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_dir = os.path.join(tmpdirname, 'new_model')
            with flagsaver.flagsaver(skip_play=True,
                                     skip_plot=True,
                                     pmap=True,
                                     batch_size=8,
                                     save_dir=model_dir):
                main.main(None)
            with flagsaver.flagsaver(skip_play=True,
                                     skip_plot=True,
                                     pmap=True,
                                     batch_size=8,
                                     load_dir=model_dir):
                main.main(None)

    @flagsaver.flagsaver(embed_model='ResNetV2Embed',
                         norm_mixed_precision_policy=
                         'params=float32,compute=float32,output=float32',
                         mixed_precision_policy=
                         'params=float32,compute=bfloat16,output=bfloat16',
                         pmap=True,
                         batch_size=8,
                         skip_play=True,
                         skip_plot=True)
    def test_resnet_trains_noexcept(self):
        os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
        self.assertEqual(jax.device_count(), 8)
        main.main(None)

    @flagsaver.flagsaver(batch_size=8,
                         training_steps=3,
                         log_training_frequency=3,
                         optimizer='adamw',
                         learning_rate=1,
                         embed_model='IdentityEmbed',
                         transition_model='RealTransition',
                         value_model='TrompTaylorValue',
                         policy_model='LinearConvPolicy')
    def test_real_linear_policy_learns_to_avoid_occupied_spaces(self):
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        all_models_build_config = models.get_all_models_build_config(
            FLAGS.board_size)
        go_model, init_params = models.build_model_with_params(
            all_models_build_config, rng_key)
        states = gojax.decode_states("""
                                    _ B _ W _
                                    W _ B _ W
                                    _ W _ B _
                                    B _ W _ B
                                    _ B _ W _
                                    """)

        trained_params, _ = manager.train_model(go_model, init_params,
                                                all_models_build_config,
                                                rng_key)

        embeddings = go_model.apply[models.EMBED_INDEX](trained_params,
                                                        rng_key, states)
        policy_logits = go_model.apply[models.POLICY_INDEX](trained_params,
                                                            rng_key,
                                                            embeddings)
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
    unittest.main()
