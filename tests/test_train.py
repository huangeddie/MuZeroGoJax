"""Tests train module."""
# pylint: disable=too-many-public-methods
import os.path
import tempfile
import unittest

import chex
import jax.numpy as jnp
from absl.testing import flagsaver

from muzero_gojax import main
from muzero_gojax import train

FLAGS = main.FLAGS


class TrainCase(chex.TestCase):
    """Tests train module."""

    def setUp(self):
        FLAGS.mark_as_parsed()

    def test_maybe_save_model_saves_model_with_bfloat16_type(self):
        """Saving bfloat16 model weights should be ok."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            with flagsaver.flagsaver(save_dir=tmpdirname, embed_model='linear_conv',
                                     value_model='linear', policy_model='linear',
                                     transition_model='linear_conv'):
                params = {'foo': jnp.array(0, dtype='bfloat16')}
                model_dir = os.path.join(tmpdirname, train.hash_model_flags(FLAGS))
                train.save_model(params, model_dir)
                self.assertTrue(os.path.exists(model_dir))

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
