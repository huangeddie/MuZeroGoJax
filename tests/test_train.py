"""Tests train module."""
# pylint: disable=too-many-public-methods
import os.path
import tempfile
import unittest

import chex
import haiku as hk
import jax.numpy as jnp
import jax.random
import numpy as np
from absl.testing import flagsaver

from muzero_gojax import main
from muzero_gojax import models
from muzero_gojax import train

FLAGS = main.FLAGS


class TrainCase(chex.TestCase):
    """Tests train module."""

    def setUp(self):
        FLAGS.mark_as_parsed()

    def test_load_model_bfloat16(self):
        """Loading bfloat16 model weights should be ok."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            with flagsaver.flagsaver(save_dir=tmpdirname, embed_model='linear_conv',
                                     value_model='linear', policy_model='linear',
                                     transition_model='linear_conv'):
                model = hk.transform(lambda x: models.value.Linear3DValue(FLAGS)(x))
                rng_key = jax.random.PRNGKey(FLAGS.rng)
                go_state = jax.random.normal(rng_key, (1024, 6, 19, 19))
                params = model.init(rng_key, go_state)
                params = jax.tree_util.tree_map(lambda x: x.astype('bfloat16'), params)
                expected_output = model.apply(params, rng_key, go_state)
                model_dir = train.maybe_save_model(params, train.hash_model_flags(FLAGS))
                params = train.load_tree_array(os.path.join(model_dir, 'params.npz'), 'bfloat16')
                np.testing.assert_array_equal(model.apply(params, rng_key, go_state),
                                              expected_output)

    def test_load_model_float32(self):
        """Loading float32 model weights should be ok."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            with flagsaver.flagsaver(save_dir=tmpdirname, embed_model='linear_conv',
                                     value_model='linear', policy_model='linear',
                                     transition_model='linear_conv'):
                model = hk.transform(lambda x: models.value.Linear3DValue(FLAGS)(x))
                rng_key = jax.random.PRNGKey(FLAGS.rng)
                go_state = jax.random.normal(rng_key, (1024, 6, 19, 19))
                params = model.init(rng_key, go_state)
                expected_output = model.apply(params, rng_key, go_state)
                model_dir = train.maybe_save_model(params, train.hash_model_flags(FLAGS))
                params = train.load_tree_array(os.path.join(model_dir, 'params.npz'), 'float32')
                np.testing.assert_allclose(model.apply(params, rng_key, go_state),
                                           expected_output.astype('float32'), rtol=0.1)

    def test_load_model_bfloat16_to_float32(self):
        """Loading float32 model weights from saved bfloat16 weights should be ok."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            with flagsaver.flagsaver(save_dir=tmpdirname, embed_model='linear_conv',
                                     value_model='linear', policy_model='linear',
                                     transition_model='linear_conv'):
                model = hk.transform(lambda x: models.value.Linear3DValue(FLAGS)(x))
                rng_key = jax.random.PRNGKey(FLAGS.rng)
                go_state = jax.random.normal(rng_key, (1024, 6, 19, 19))
                params = model.init(rng_key, go_state)
                params = jax.tree_util.tree_map(lambda x: x.astype('bfloat16'), params)
                expected_output = model.apply(params, rng_key, go_state)
                model_dir = train.maybe_save_model(params, train.hash_model_flags(FLAGS))
                params = train.load_tree_array(os.path.join(model_dir, 'params.npz'), 'float32')
                np.testing.assert_allclose(model.apply(params, rng_key, go_state),
                                           expected_output.astype('float32'), rtol=0.1)

    def test_load_model_float32_to_bfloat16_approximation(self):
        """Loading float32 model weights from bfloat16 should be ok with some inconsistencies."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            with flagsaver.flagsaver(save_dir=tmpdirname, embed_model='linear_conv',
                                     value_model='linear', policy_model='linear',
                                     transition_model='linear_conv'):
                model = hk.transform(lambda x: models.value.Linear3DValue(FLAGS)(x))
                rng_key = jax.random.PRNGKey(FLAGS.rng)
                go_state = jax.random.normal(rng_key, (1024, 6, 19, 19))
                params = model.init(rng_key, go_state)
                expected_output = model.apply(params, rng_key, go_state)
                model_dir = train.maybe_save_model(params, train.hash_model_flags(FLAGS))
                params = train.load_tree_array(os.path.join(model_dir, 'params.npz'), 'bfloat16')
                np.testing.assert_allclose(model.apply(params, rng_key, go_state),
                                           expected_output.astype('float32'), rtol=1)

    def test_maybe_save_model_saves_model_with_bfloat16_type(self):
        """Saving bfloat16 model weights should be ok."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            with flagsaver.flagsaver(save_dir=tmpdirname, embed_model='linear_conv',
                                     value_model='linear', policy_model='linear',
                                     transition_model='linear_conv'):
                params = {'foo': jnp.array(0, dtype='bfloat16')}
                model_dir = train.maybe_save_model(params, train.hash_model_flags(FLAGS))
                self.assertTrue(os.path.exists(model_dir))

    @flagsaver.flagsaver
    def test_maybe_save_model_empty_save_dir(self):
        """No save should return empty."""
        params = {}
        self.assertIsNone(train.maybe_save_model(params, train.hash_model_flags(FLAGS)))

    def test_hash_flags_invariant_to_load_dir(self):
        """Hash of flags should be invariant to load_dir."""
        with flagsaver.flagsaver(load_dir='foo'):
            expected_hash = train.hash_model_flags(FLAGS)
        with flagsaver.flagsaver(load_dir='bar'):
            self.assertEqual(train.hash_model_flags(FLAGS), expected_hash)

    def test_hash_flags_changes_with_embed_model(self):
        """Hash of flags should vary with embed model name."""
        with flagsaver.flagsaver(embed_model='linear_conv'):
            expected_hash = train.hash_model_flags(FLAGS)
        with flagsaver.flagsaver(embed_model='cnn_lite'):
            self.assertNotEqual(train.hash_model_flags(FLAGS), expected_hash)

    def test_hash_flags_changes_with_hdim(self):
        """Hash of flags should vary with hdim."""
        with flagsaver.flagsaver(hdim=8):
            expected_hash = train.hash_model_flags(FLAGS)
        with flagsaver.flagsaver(hdim=32):
            self.assertNotEqual(train.hash_model_flags(FLAGS), expected_hash)


if __name__ == '__main__':
    unittest.main()
