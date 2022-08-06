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
        model_state = {}
        self.assertIsNone(train.maybe_save_model(params, model_state, main.FLAGS))

    def test_maybe_save_model_saves_model_with_bfloat16_type(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            main.FLAGS(f'foo --save_dir={tmpdirname} --embed_model=linear --value_model=linear '
                       f'--policy_model=linear --transition_model=linear'.split())
            params = {'foo': jnp.array(0, dtype='bfloat16')}
            model_state = {'bar': jnp.array(0, dtype='bfloat16')}
            model_dir = train.maybe_save_model(params, model_state, main.FLAGS)
            self.assertTrue(os.path.exists(model_dir))

    def test_hash_flags_invariant_to_load_dir(self):
        main.FLAGS(f'foo --load_dir=foo'.split())
        expected_hash = train.hash_model_flags(main.FLAGS)
        main.FLAGS.unparse_flags()
        main.FLAGS(f'foo --load_dir=bar'.split())
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
            model = hk.transform_with_state(lambda x: models.value.Linear3DValue(main.FLAGS.board_size, hdim=None)(x))
            rng_key = jax.random.PRNGKey(main.FLAGS.random_seed)
            go_state = jax.random.normal(rng_key, (1024, 6, 19, 19))
            params, model_state = model.init(rng_key, go_state)
            params = jax.tree_util.tree_map(lambda x: x.astype('bfloat16'), params)
            expected_output, _ = model.apply(params, model_state, rng_key, go_state)
            model_dir = train.maybe_save_model(params, model_state, main.FLAGS)
            params = train.load_tree_array(os.path.join(model_dir, 'params.npz'), 'bfloat16')
            np.testing.assert_array_equal(model.apply(params, model_state, rng_key, go_state)[0], expected_output)

    def test_load_model_float32(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            main.FLAGS(f'foo --save_dir={tmpdirname} --embed_model=linear --value_model=linear '
                       f'--policy_model=linear --transition_model=linear'.split())
            model = hk.transform_with_state(lambda x: models.value.Linear3DValue(main.FLAGS.board_size, hdim=None)(x))
            rng_key = jax.random.PRNGKey(main.FLAGS.random_seed)
            go_state = jax.random.normal(rng_key, (1024, 6, 19, 19))
            params, model_state = model.init(rng_key, go_state)
            expected_output, _ = model.apply(params, model_state, rng_key, go_state)
            model_dir = train.maybe_save_model(params, model_state, main.FLAGS)
            params = train.load_tree_array(os.path.join(model_dir, 'params.npz'), 'float32')
            np.testing.assert_allclose(model.apply(params, model_state, rng_key, go_state)[0],
                                       expected_output.astype('float32'), rtol=0.1)

    def test_load_model_bfloat16_to_float32(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            main.FLAGS(f'foo --save_dir={tmpdirname} --embed_model=linear --value_model=linear '
                       f'--policy_model=linear --transition_model=linear'.split())
            model = hk.transform_with_state(lambda x: models.value.Linear3DValue(main.FLAGS.board_size, hdim=None)(x))
            rng_key = jax.random.PRNGKey(main.FLAGS.random_seed)
            go_state = jax.random.normal(rng_key, (1024, 6, 19, 19))
            params, model_state = model.init(rng_key, go_state)
            params = jax.tree_util.tree_map(lambda x: x.astype('bfloat16'), params)
            expected_output, _ = model.apply(params, model_state, rng_key, go_state)
            model_dir = train.maybe_save_model(params, model_state, main.FLAGS)
            params = train.load_tree_array(os.path.join(model_dir, 'params.npz'), 'float32')
            np.testing.assert_allclose(model.apply(params, model_state, rng_key, go_state)[0],
                                       expected_output.astype('float32'), rtol=0.1)

    def test_load_model_float32_to_bfloat16_approximation(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            main.FLAGS(f'foo --save_dir={tmpdirname} --embed_model=linear --value_model=linear '
                       f'--policy_model=linear --transition_model=linear'.split())
            model = hk.transform_with_state(lambda x: models.value.Linear3DValue(main.FLAGS.board_size, hdim=None)(x))
            rng_key = jax.random.PRNGKey(main.FLAGS.random_seed)
            go_state = jax.random.normal(rng_key, (1024, 6, 19, 19))
            params, model_state = model.init(rng_key, go_state)
            expected_output, _ = model.apply(params, model_state, rng_key, go_state)
            model_dir = train.maybe_save_model(params, model_state, main.FLAGS)
            params = train.load_tree_array(os.path.join(model_dir, 'params.npz'), 'bfloat16')
            np.testing.assert_allclose(model.apply(params, model_state, rng_key, go_state)[0],
                                       expected_output.astype('float32'), rtol=1)


if __name__ == '__main__':
    unittest.main()
