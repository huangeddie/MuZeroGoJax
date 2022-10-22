"""Tests train module."""
# pylint: disable=too-many-public-methods,missing-function-docstring
import unittest

import chex
from absl.testing import flagsaver
from absl.testing import parameterized

from muzero_gojax import main
from muzero_gojax import train

FLAGS = main.FLAGS


class TrainCase(chex.TestCase):
    """Tests train module."""

    def setUp(self):
        FLAGS.mark_as_parsed()

    def test_hash_flags_is_invariant_to_load_dir(self):
        with flagsaver.flagsaver(load_dir='foo'):
            expected_hash = train.hash_model_flags(FLAGS)
        with flagsaver.flagsaver(load_dir='bar'):
            self.assertEqual(train.hash_model_flags(FLAGS), expected_hash)

    @parameterized.named_parameters(('embed_model', 'embed_model'),
                                    ('decode_model', 'decode_model'),
                                    ('value_model', 'value_model'),
                                    ('policy_model', 'policy_model'),
                                    ('transition_model', 'transition_model'), )
    def test_hash_flags_changes_with_model_flags(self, model_flag):
        with flagsaver.flagsaver(**{model_flag: 'foo'}):
            expected_hash = train.hash_model_flags(FLAGS)
        with flagsaver.flagsaver(**{model_flag: 'bar'}):
            self.assertNotEqual(train.hash_model_flags(FLAGS), expected_hash)

    def test_hash_flags_changes_with_hdim(self):
        with flagsaver.flagsaver(hdim=8):
            expected_hash = train.hash_model_flags(FLAGS)
        with flagsaver.flagsaver(hdim=32):
            self.assertNotEqual(train.hash_model_flags(FLAGS), expected_hash)

    def test_hash_flags_changes_with_embed_dim(self):
        with flagsaver.flagsaver(embed_dim=8):
            expected_hash = train.hash_model_flags(FLAGS)
        with flagsaver.flagsaver(embed_dim=32):
            self.assertNotEqual(train.hash_model_flags(FLAGS), expected_hash)


if __name__ == '__main__':
    unittest.main()
