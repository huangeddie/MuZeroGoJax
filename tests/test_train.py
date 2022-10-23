"""Tests train module."""
# pylint: disable=too-many-public-methods,missing-function-docstring
import unittest

import chex
from absl.testing import flagsaver
from absl.testing import parameterized

from muzero_gojax import main
from muzero_gojax import models
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

    @flagsaver.flagsaver(training_steps=1, board_size=3)
    def test_train_model_raises_no_error_with_basic_use_case(self):
        go_model, params = models.make_model(board_size=FLAGS.board_size)
        _, _ = train.train_model(go_model, params, board_size=FLAGS.board_size)


if __name__ == '__main__':
    unittest.main()
