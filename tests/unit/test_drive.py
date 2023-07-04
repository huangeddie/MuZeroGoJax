"""Tests for the drive module."""
# pylint: disable=missing-function-docstring, missing-class-docstring
import os
import tempfile
import unittest

from absl.testing import absltest, flagsaver

from muzero_gojax import drive, main

FLAGS = main.FLAGS


class DriveTestCase(unittest.TestCase):

    def setUp(self):
        FLAGS.mark_as_parsed()

    def test_drive_creates_directory(self):
        with tempfile.TemporaryDirectory() as tempdir:
            save_dir = os.path.join(tempdir, 'foo')
            self.assertFalse(drive.directory_exists(save_dir))
            with flagsaver.flagsaver(save_dir=save_dir):
                drive.initialize_drive(save_dir, FLAGS)
            self.assertTrue(drive.directory_exists(save_dir))

    def test_drive_saves_flags(self):
        with tempfile.TemporaryDirectory() as save_dir:
            drive.initialize_drive(save_dir, FLAGS)
            self.assertTrue(os.path.exists(os.path.join(save_dir, 'flags.txt')))


if __name__ == '__main__':
    absltest.main()