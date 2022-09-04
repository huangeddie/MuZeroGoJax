"""Tests train module."""
import functools
import os.path
import tempfile
import unittest

import chex
import gojax
import haiku as hk
import jax.numpy as jnp
import jax.random
import numpy as np

from muzero_gojax import main
from muzero_gojax import models
from muzero_gojax import train


def test_load_model_bfloat16():
    """Loading bfloat16 model weights should be ok."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        main.FLAGS.unparse_flags()
        main.FLAGS(f'foo --save_dir={tmpdirname} --embed_model=linear_conv --value_model=linear '
                   f'--policy_model=linear --transition_model=linear_conv'.split())
        model = hk.transform(lambda x: models.value.Linear3DValue(main.FLAGS)(x))
        rng_key = jax.random.PRNGKey(main.FLAGS.rng)
        go_state = jax.random.normal(rng_key, (1024, 6, 19, 19))
        params = model.init(rng_key, go_state)
        params = jax.tree_util.tree_map(lambda x: x.astype('bfloat16'), params)
        expected_output = model.apply(params, rng_key, go_state)
        model_dir = train.maybe_save_model(params, main.FLAGS)
        params = train.load_tree_array(os.path.join(model_dir, 'params.npz'), 'bfloat16')
        np.testing.assert_array_equal(model.apply(params, rng_key, go_state), expected_output)


def test_load_model_float32():
    """Loading float32 model weights should be ok."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        main.FLAGS.unparse_flags()
        main.FLAGS(f'foo --save_dir={tmpdirname} --embed_model=linear_conv --value_model=linear '
                   f'--policy_model=linear --transition_model=linear_conv'.split())
        model = hk.transform(lambda x: models.value.Linear3DValue(main.FLAGS)(x))
        rng_key = jax.random.PRNGKey(main.FLAGS.rng)
        go_state = jax.random.normal(rng_key, (1024, 6, 19, 19))
        params = model.init(rng_key, go_state)
        expected_output = model.apply(params, rng_key, go_state)
        model_dir = train.maybe_save_model(params, main.FLAGS)
        params = train.load_tree_array(os.path.join(model_dir, 'params.npz'), 'float32')
        np.testing.assert_allclose(model.apply(params, rng_key, go_state),
                                   expected_output.astype('float32'), rtol=0.1)


def test_load_model_bfloat16_to_float32():
    """Loading float32 model weights from saved bfloat16 weights should be ok."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        main.FLAGS.unparse_flags()
        main.FLAGS(f'foo --save_dir={tmpdirname} --embed_model=linear_conv --value_model=linear '
                   f'--policy_model=linear --transition_model=linear_conv'.split())
        model = hk.transform(lambda x: models.value.Linear3DValue(main.FLAGS)(x))
        rng_key = jax.random.PRNGKey(main.FLAGS.rng)
        go_state = jax.random.normal(rng_key, (1024, 6, 19, 19))
        params = model.init(rng_key, go_state)
        params = jax.tree_util.tree_map(lambda x: x.astype('bfloat16'), params)
        expected_output = model.apply(params, rng_key, go_state)
        model_dir = train.maybe_save_model(params, main.FLAGS)
        params = train.load_tree_array(os.path.join(model_dir, 'params.npz'), 'float32')
        np.testing.assert_allclose(model.apply(params, rng_key, go_state),
                                   expected_output.astype('float32'), rtol=0.1)


def test_load_model_float32_to_bfloat16_approximation():
    """Loading float32 model weights from bfloat16 should be ok with some inconsistencies."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        main.FLAGS.unparse_flags()
        main.FLAGS(f'foo --save_dir={tmpdirname} --embed_model=linear_conv --value_model=linear '
                   f'--policy_model=linear --transition_model=linear_conv'.split())
        model = hk.transform(lambda x: models.value.Linear3DValue(main.FLAGS)(x))
        rng_key = jax.random.PRNGKey(main.FLAGS.rng)
        go_state = jax.random.normal(rng_key, (1024, 6, 19, 19))
        params = model.init(rng_key, go_state)
        expected_output = model.apply(params, rng_key, go_state)
        model_dir = train.maybe_save_model(params, main.FLAGS)
        params = train.load_tree_array(os.path.join(model_dir, 'params.npz'), 'bfloat16')
        np.testing.assert_allclose(model.apply(params, rng_key, go_state),
                                   expected_output.astype('float32'), rtol=1)


class TrainCase(chex.TestCase):
    """Tests train module."""

    def test_maybe_save_model_empty_save_dir(self):
        """No save should return empty."""
        main.FLAGS.unparse_flags()
        main.FLAGS([''])
        params = {}
        self.assertIsNone(train.maybe_save_model(params, main.FLAGS))

    def test_maybe_save_model_saves_model_with_bfloat16_type(self):
        """Saving bfloat16 model weights should be ok."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            main.FLAGS.unparse_flags()
            main.FLAGS(
                f'foo --save_dir={tmpdirname} --embed_model=linear_conv --value_model=linear '
                f'--policy_model=linear --transition_model=linear_conv'.split())
            params = {'foo': jnp.array(0, dtype='bfloat16')}
            model_dir = train.maybe_save_model(params, main.FLAGS)
            self.assertTrue(os.path.exists(model_dir))

    def test_hash_flags_invariant_to_load_dir(self):
        """Hash of flags should be invariant to load_dir."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --load_dir=foo'.split())
        expected_hash = train.hash_model_flags(main.FLAGS)
        main.FLAGS.unparse_flags()
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --load_dir=bar'.split())
        self.assertEqual(train.hash_model_flags(main.FLAGS), expected_hash)

    def test_hash_flags_changes_with_embed_model(self):
        """Hash of flags should vary with embed model name."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --embed_model=linear_conv'.split())
        expected_hash = train.hash_model_flags(main.FLAGS)
        main.FLAGS.unparse_flags()
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --embed_model=cnn_lite'.split())
        self.assertNotEqual(train.hash_model_flags(main.FLAGS), expected_hash)

    def test_hash_flags_changes_with_hdim(self):
        """Hash of flags should vary with hdim."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --hdim=8'.split())
        expected_hash = train.hash_model_flags(main.FLAGS)
        main.FLAGS.unparse_flags()
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --hdim=32'.split())
        self.assertNotEqual(train.hash_model_flags(main.FLAGS), expected_hash)

    def test_compute_loss_gradients_yields_negative_value_gradients(self):
        """
        Given a model with positive parameters and a single won state, check that the value
        parameter gradients are negative.
        """
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --hdim=2 --embed_model=linear_conv --value_model=linear '
                   '--policy_model=linear --transition_model=linear_conv --hypo_steps=1'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        params = jax.tree_util.tree_map(lambda x: jnp.full_like(x, 1e-3), params)
        nt_states = gojax.decode_states("""
                                            _ _ _
                                            _ B _
                                            _ _ _
                                            """)
        trajectories = {
            'nt_states': jnp.reshape(nt_states, (1, 1, 6, 3, 3)),
            'nt_actions': jnp.full((1, 1), fill_value=-1, dtype='uint16')
        }
        grads, _ = train.compute_loss_gradients(main.FLAGS, go_model, params, trajectories)
        self.assertIn('linear3_d_value', grads)
        self.assertIn('value_w', grads['linear3_d_value'])
        self.assertIn('value_b', grads['linear3_d_value'])
        np.testing.assert_array_less(grads['linear3_d_value']['value_w'],
                                     jnp.zeros_like(grads['linear3_d_value']['value_w']))
        np.testing.assert_array_less(grads['linear3_d_value']['value_b'],
                                     jnp.zeros_like(grads['linear3_d_value']['value_b']))

    def test_compute_loss_gradients_yields_positive_value_gradients(self):
        """
        Given a model with positive parameters and a single loss state, check that the value
        parameter gradients are positive.
        """
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --hdim=2 --embed_model=linear_conv --value_model=linear '
                   '--policy_model=linear --transition_model=linear_conv --hypo_steps=1'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        params = jax.tree_util.tree_map(lambda x: jnp.full_like(x, 1e-3), params)
        nt_states = gojax.decode_states("""
                                            _ _ _
                                            _ W _
                                            _ _ _
                                            """)
        trajectories = {
            'nt_states': jnp.reshape(nt_states, (1, 1, 6, 3, 3)),
            'nt_actions': jnp.full((1, 1), fill_value=-1, dtype='uint16')
        }
        grads, _ = train.compute_loss_gradients(main.FLAGS, go_model, params, trajectories)
        self.assertIn('linear3_d_value', grads)
        self.assertIn('value_w', grads['linear3_d_value'])
        self.assertIn('value_b', grads['linear3_d_value'])
        np.testing.assert_array_less(-grads['linear3_d_value']['value_w'],
                                     jnp.zeros_like(grads['linear3_d_value']['value_w']))
        np.testing.assert_array_less(-grads['linear3_d_value']['value_b'],
                                     jnp.zeros_like(grads['linear3_d_value']['value_b']))

    def test_compute_loss_gradients_with_one_step_has_nonzero_grads_except_for_transition(self):
        """Tests all parameters except for transitions have grads with compute_0_step_total_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --hdim=2 --embed_model=linear_conv --value_model=linear '
                   '--policy_model=linear --transition_model=linear_conv --hypo_steps=1'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        trajectories = {
            'nt_states': jnp.ones((1, 1, 6, 3, 3), dtype=bool),
            'nt_actions': jnp.ones((1, 1), dtype='uint16')
        }
        grads, _ = train.compute_loss_gradients(main.FLAGS, go_model, params, trajectories)

        # Check all transition weights are 0.
        self.assertFalse(grads['linear_conv_transition/~/conv2_d']['b'].astype(bool).any())
        self.assertFalse(grads['linear_conv_transition/~/conv2_d']['w'].astype(bool).any())
        # Check everything else is non-zero.
        del grads['linear_conv_transition/~/conv2_d']['b']
        del grads['linear_conv_transition/~/conv2_d']['w']
        self.assertTrue(functools.reduce(lambda a, b: a and b,
                                         map(lambda grad: grad.astype(bool).all(),
                                             jax.tree_util.tree_flatten(grads)[0])))

    def test_compute_loss_gradients_with_two_steps_has_nonzero_grads(self):
        """Tests all parameters except for transitions have grads with compute_0_step_total_loss."""
        main.FLAGS.unparse_flags()
        main.FLAGS('foo --board_size=3 --hdim=2 --embed_model=linear_conv --value_model=linear '
                   '--policy_model=linear --transition_model=linear_conv --hypo_steps=2 '
                   '--add_trans_loss=true'.split())
        go_model = models.make_model(main.FLAGS)
        params = go_model.init(jax.random.PRNGKey(42), states=jnp.ones((1, 6, 3, 3), dtype=bool))
        trajectories = {
            'nt_states': jnp.ones((1, 2, 6, 3, 3), dtype=bool),
            'nt_actions': jnp.ones((1, 2), dtype='uint16')
        }
        grads, _ = train.compute_loss_gradients(main.FLAGS, go_model, params, trajectories)

        # Check some transition weights are non-zero.
        self.assertTrue(grads['linear_conv_transition/~/conv2_d']['b'].astype(bool).any())
        self.assertTrue(grads['linear_conv_transition/~/conv2_d']['w'].astype(bool).any())
        # Check everything else is non-zero.
        del grads['linear_conv_transition/~/conv2_d']['b']
        del grads['linear_conv_transition/~/conv2_d']['w']
        self.assertTrue(functools.reduce(lambda a, b: a and b,
                                         map(lambda grad: grad.astype(bool).all(),
                                             jax.tree_util.tree_flatten(grads)[0])))


if __name__ == '__main__':
    unittest.main()
