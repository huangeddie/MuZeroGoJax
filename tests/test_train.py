"""Test miscellaneous functions in train_model.py."""
# pylint: disable=no-self-use,no-value-for-parameter,duplicate-code
import unittest

import chex
import numpy as np
from absl.testing import parameterized

from muzero_gojax import train


class TrainTestCase(chex.TestCase):
    """Test miscellaneous, functions in test.py."""

    @parameterized.named_parameters(('zero', 1, 1, 0, [[False]]), ('one', 1, 1, 1, [[True]]),
                                    ('zeros', 1, 2, 0, [[False, False]]),
                                    ('half', 1, 2, 1, [[True, False]]),
                                    ('full', 1, 2, 2, [[True, True]]),
                                    ('b2_zero', 2, 1, 0, [[False], [False]]),
                                    ('b2_one', 2, 1, 1, [[True], [True]]),
                                    ('b2_zeros', 2, 2, 0, [[False, False], [False, False]]),
                                    ('b2_half', 2, 2, 1, [[True, False], [True, False]]),
                                    ('b2_full', 2, 2, 2, [[True, True], [True, True]]), )
    def test_k_steps_mask_(self, batch_size, total_steps, k, expected_output):
        """Tests the make_first_k_steps_mask based on inputs and expected output."""
        np.testing.assert_array_equal(train.make_first_k_steps_mask(batch_size, total_steps, k),
                                      expected_output)


if __name__ == '__main__':
    unittest.main()
