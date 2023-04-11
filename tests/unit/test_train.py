"""Tests train module."""
# pylint: disable=too-many-public-methods,missing-function-docstring
import unittest

import chex

from muzero_gojax import main

FLAGS = main.FLAGS


class TrainCase(chex.TestCase):
    """Tests train module."""

    def setUp(self):
        FLAGS.mark_as_parsed()

    # TODO: Add tests.


if __name__ == '__main__':
    unittest.main()
