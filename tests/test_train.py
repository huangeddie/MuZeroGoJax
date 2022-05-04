"""Tests train.py."""
# pylint: disable=missing-function-docstring,no-self-use,unnecessary-lambda
import unittest

import haiku as hk
import jax

import models
import train


class TrainTestCase(unittest.TestCase):
    """Tests train.py."""

    def test_train_random_model_parameters_structure(self):
        go_model = hk.transform(lambda states: models.RandomGoModel()(states))
        params = train.train(go_model, batch_size=2, board_size=3, training_steps=1,
                             max_num_steps=1, rng_key=jax.random.PRNGKey(42))
        self.assertIsInstance(params, dict)
        self.assertEqual(len(params), 0)


if __name__ == '__main__':
    unittest.main()
