import os.path
import tempfile
import unittest

import chex
import haiku as hk
import jax.numpy as jnp
import jax.random
import numpy as np

from muzero_gojax import main
from muzero_gojax import models
from muzero_gojax import train


class TrainCase(chex.TestCase):
    def setUp(self):
        main.FLAGS.unparse_flags()

    def test_maybe_save_model_no_save(self):
        main.FLAGS([''])
        params = {}
        self.assertIsNone(train.maybe_save_model(params, main.FLAGS))

    def test_maybe_save_model_saves_model_with_bfloat16_type(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            main.FLAGS(f'foo --save_dir={tmpdirname} --embed_model=linear --value_model=linear '
                       f'--policy_model=linear --transition_model=linear'.split())
            params = {'foo': jnp.array(0, dtype='bfloat16')}
            filename = train.maybe_save_model(params, main.FLAGS)
            self.assertTrue(os.path.exists(filename))

    def test_hash_flags_invariant_to_load_path(self):
        main.FLAGS(f'foo --load_path=foo'.split())
        expected_hash = train.hash_model_flags(main.FLAGS)
        main.FLAGS.unparse_flags()
        main.FLAGS(f'foo --load_path=bar'.split())
        self.assertEqual(train.hash_model_flags(main.FLAGS), expected_hash)

    def test_hash_flags_changes_with_embed_model(self):
        main.FLAGS(f'foo --embed_model=linear'.split())
        expected_hash = train.hash_model_flags(main.FLAGS)
        main.FLAGS.unparse_flags()
        main.FLAGS(f'foo --embed_model=cnn_lite'.split())
        self.assertNotEqual(train.hash_model_flags(main.FLAGS), expected_hash)

    def test_hash_flags_changes_with_hdim(self):
        main.FLAGS(f'foo --hdim=8'.split())
        expected_hash = train.hash_model_flags(main.FLAGS)
        main.FLAGS.unparse_flags()
        main.FLAGS(f'foo --hdim=32'.split())
        self.assertNotEqual(train.hash_model_flags(main.FLAGS), expected_hash)

    def test_load_model_bfloat16(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            main.FLAGS(f'foo --save_dir={tmpdirname} --embed_model=linear --value_model=linear '
                       f'--policy_model=linear --transition_model=linear'.split())
            model = hk.transform(lambda x: models.value.Linear3DValue(main.FLAGS.board_size, hdim=None)(x))
            rng_key = jax.random.PRNGKey(main.FLAGS.random_seed)
            go_state = jax.random.normal(rng_key, (1024, 6, 19, 19))
            params = jax.tree_util.tree_map(lambda x: x.astype('bfloat16'), model.init(rng_key, go_state))
            expected_output = model.apply(params, rng_key, go_state)
            filename = train.maybe_save_model(params, main.FLAGS)
            params = train.load_params(filename, 'bfloat16')
            np.testing.assert_array_equal(model.apply(params, rng_key, go_state), expected_output)

    def test_load_model_float32(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            main.FLAGS(f'foo --save_dir={tmpdirname} --embed_model=linear --value_model=linear '
                       f'--policy_model=linear --transition_model=linear'.split())
            model = hk.transform(lambda x: models.value.Linear3DValue(main.FLAGS.board_size, hdim=None)(x))
            rng_key = jax.random.PRNGKey(main.FLAGS.random_seed)
            go_state = jax.random.normal(rng_key, (1024, 6, 19, 19))
            params = model.init(rng_key, go_state)
            expected_output = model.apply(params, rng_key, go_state)
            filename = train.maybe_save_model(params, main.FLAGS)
            params = train.load_params(filename, 'float32')
            np.testing.assert_allclose(model.apply(params, rng_key, go_state), expected_output.astype('float32'),
                                       rtol=0.1)

    def test_load_model_bfloat16_to_float32(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            main.FLAGS(f'foo --save_dir={tmpdirname} --embed_model=linear --value_model=linear '
                       f'--policy_model=linear --transition_model=linear'.split())
            model = hk.transform(lambda x: models.value.Linear3DValue(main.FLAGS.board_size, hdim=None)(x))
            rng_key = jax.random.PRNGKey(main.FLAGS.random_seed)
            go_state = jax.random.normal(rng_key, (1024, 6, 19, 19))
            params = jax.tree_util.tree_map(lambda x: x.astype('bfloat16'), model.init(rng_key, go_state))
            expected_output = model.apply(params, rng_key, go_state)
            filename = train.maybe_save_model(params, main.FLAGS)
            params = train.load_params(filename, 'float32')
            np.testing.assert_allclose(model.apply(params, rng_key, go_state), expected_output.astype('float32'),
                                       rtol=0.1)

    def test_load_model_float32_to_bfloat16_approximation(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            main.FLAGS(f'foo --save_dir={tmpdirname} --embed_model=linear --value_model=linear '
                       f'--policy_model=linear --transition_model=linear'.split())
            model = hk.transform(lambda x: models.value.Linear3DValue(main.FLAGS.board_size, hdim=None)(x))
            rng_key = jax.random.PRNGKey(main.FLAGS.random_seed)
            go_state = jax.random.normal(rng_key, (1024, 6, 19, 19))
            params = model.init(rng_key, go_state)
            expected_output = model.apply(params, rng_key, go_state)
            filename = train.maybe_save_model(params, main.FLAGS)
            params = train.load_params(filename, 'bfloat16')
            np.testing.assert_allclose(model.apply(params, rng_key, go_state), expected_output.astype('float32'),
                                       rtol=1)


if __name__ == '__main__':
    unittest.main()
