"""Tests losses.py."""
# pylint: disable=missing-function-docstring,no-value-for-parameter,too-many-public-methods,duplicate-code,unnecessary-lambda
import chex
import gojax
import jax
import jax.random
import numpy as np
import optax
from absl.testing import absltest, flagsaver
from jax import numpy as jnp

from muzero_gojax import game, losses, main, models, nt_utils

FLAGS = main.FLAGS


def _ones_like_game_data(board_size: int, batch_size: int,
                         hypo_steps: int) -> losses.GameData:
    nk_states = nt_utils.unflatten_first_dim(
        jnp.ones_like(
            gojax.new_states(board_size, batch_size * (hypo_steps + 1))),
        batch_size, (hypo_steps + 1))
    return losses.GameData(nk_states=nk_states,
                           nk_actions=jnp.ones((batch_size, (hypo_steps + 1)),
                                               dtype='uint16'),
                           nk_player_labels=-jnp.ones(
                               (batch_size, (hypo_steps + 1)), dtype='int8'))


def _small_3x3_linear_model_flags():
    return {
        'board_size': 3,
        'hdim': 2,
        'embed_nlayers': 0,
        'value_nlayers': 0,
        'decode_nlayers': 0,
        'policy_nlayers': 0,
        'transition_nlayers': 0,
        'embed_model': 'NonSpatialConvEmbed',
        'value_model': 'NonSpatialConvValue',
        'decode_model': 'NonSpatialConvDecode',
        'policy_model': 'NonSpatialConvPolicy',
        'transition_model': 'NonSpatialConvTransition'
    }


class ComputeLossGradientsAndMetricsTestCase(chex.TestCase):
    """Test losses.py"""

    def assert_tree_leaves_all_zero(self, pytree: chex.ArrayTree):
        chex.assert_trees_all_equal(
            pytree, jax.tree_util.tree_map(lambda x: jnp.zeros_like(x),
                                           pytree))

    def assert_tree_leaves_any_non_zero(self, pytree: chex.ArrayTree):
        if not jax.tree_util.tree_reduce(
                lambda any_non_zero, leaf: any_non_zero or jnp.any(leaf),
                pytree,
                initializer=False):
            self.fail("")

    def assert_tree_leaves_all_non_zero(self, pytree: chex.ArrayTree):
        binary_pytree = jax.tree_util.tree_map(lambda x: x.astype(bool),
                                               pytree)
        chex.assert_trees_all_equal(
            binary_pytree,
            jax.tree_util.tree_map(lambda x: jnp.ones_like(x), binary_pytree))

    def assert_tree_leaves_all_close_to_zero(self, pytree: chex.ArrayTree,
                                             atol: float):
        chex.assert_trees_all_close(pytree,
                                    jax.tree_util.tree_map(
                                        lambda x: jnp.zeros_like(x), pytree),
                                    atol=atol)

    def test_assert_tree_leaves_all_zero(self):
        self.assert_tree_leaves_all_zero({
            'a': jnp.zeros(()),
            'b': {
                'c': jnp.zeros(2)
            }
        })
        with self.assertRaises(AssertionError):
            self.assert_tree_leaves_all_zero({
                'a': jnp.zeros(()),
                'b': {
                    'c': jnp.array([0, 1])
                }
            })

    def test_assert_tree_leaves_any_non_zero(self):
        self.assert_tree_leaves_any_non_zero({
            'a': jnp.zeros(()),
            'b': {
                'c': jnp.array([0, 1])
            }
        })
        with self.assertRaises(AssertionError):
            self.assert_tree_leaves_any_non_zero({
                'a': jnp.zeros(()),
                'b': {
                    'c': jnp.zeros(2)
                }
            })

    def test_assert_tree_leaves_all_non_zero(self):
        self.assert_tree_leaves_all_non_zero({
            'a': jnp.ones(()),
            'b': {
                'c': jnp.ones(2)
            }
        })
        with self.assertRaises(AssertionError):
            self.assert_tree_leaves_all_non_zero({
                'a': jnp.ones(()),
                'b': {
                    'c': jnp.array([0, 1])
                }
            })

    def test_assert_pytree_allclose(self):
        self.assert_tree_leaves_all_close_to_zero(
            {
                'a': jnp.array(1e-7),
                'b': {
                    'c': jnp.array(1e-5)
                }
            }, atol=1e-4)
        with self.assertRaises(AssertionError):
            self.assert_tree_leaves_all_close_to_zero(
                {
                    'a': jnp.array(0.5),
                    'b': {
                        'c': jnp.array(1e-5)
                    }
                }, atol=1e-4)

    def setUp(self):
        FLAGS.mark_as_parsed()

    @flagsaver.flagsaver(**_small_3x3_linear_model_flags(),
                         add_value_loss=False,
                         add_hypo_value_loss=False,
                         add_decode_loss=False,
                         add_hypo_decode_loss=False,
                         add_policy_loss=False)
    def test_no_loss_returns_no_gradients(self):
        all_models_build_config = models.get_all_models_build_config(
            FLAGS.board_size, FLAGS.dtype)
        go_model, params = models.build_model_with_params(
            all_models_build_config, jax.random.PRNGKey(FLAGS.rng))
        params = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), params)
        game_data = _ones_like_game_data(FLAGS.board_size,
                                         FLAGS.batch_size,
                                         hypo_steps=1)

        rng_key = jax.random.PRNGKey(FLAGS.rng)
        grads, _ = losses.compute_loss_gradients_and_metrics(
            go_model, params, game_data, rng_key)
        self.assert_tree_leaves_all_zero(grads)

    @flagsaver.flagsaver(**_small_3x3_linear_model_flags(),
                         add_value_loss=True,
                         add_decode_loss=True,
                         add_policy_loss=True)
    def test_all_loss_gradients_are_finite(self):
        all_models_build_config = models.get_all_models_build_config(
            FLAGS.board_size, FLAGS.dtype)
        go_model, params = models.build_model_with_params(
            all_models_build_config, jax.random.PRNGKey(FLAGS.rng))
        params = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), params)
        game_data = _ones_like_game_data(FLAGS.board_size,
                                         FLAGS.batch_size,
                                         hypo_steps=1)

        rng_key = jax.random.PRNGKey(FLAGS.rng)
        grads, _ = losses.compute_loss_gradients_and_metrics(
            go_model, params, game_data, rng_key)
        chex.assert_tree_all_finite(grads)

    @flagsaver.flagsaver(**_small_3x3_linear_model_flags(),
                         add_value_loss=False,
                         add_hypo_value_loss=False,
                         add_decode_loss=False,
                         add_hypo_decode_loss=False,
                         add_policy_loss=True)
    def test_policy_loss_only_affects_embed_and_policy_gradients(self):
        all_models_build_config = models.get_all_models_build_config(
            FLAGS.board_size, FLAGS.dtype)
        go_model, params = models.build_model_with_params(
            all_models_build_config, jax.random.PRNGKey(FLAGS.rng))
        params = jax.tree_util.tree_map(
            lambda x: jax.random.normal(
                jax.random.PRNGKey(42), x.shape, dtype=FLAGS.dtype), params)
        game_data = _ones_like_game_data(FLAGS.board_size,
                                         FLAGS.batch_size,
                                         hypo_steps=1)

        grads: dict
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        grads, _ = losses.compute_loss_gradients_and_metrics(
            go_model, params, game_data, rng_key)

        self.assert_tree_leaves_any_non_zero(
            grads['non_spatial_conv_embed/~/non_spatial_conv/~/conv2_d'])
        self.assert_tree_leaves_any_non_zero(
            grads['non_spatial_conv_policy/~/non_spatial_conv/~/conv2_d'])
        self.assert_tree_leaves_any_non_zero(
            grads['non_spatial_conv_policy/~/non_spatial_conv_1/~/conv2_d'])
        # Check the remaining gradients are zero.
        grads.pop('non_spatial_conv_embed/~/non_spatial_conv/~/conv2_d')
        grads.pop('non_spatial_conv_policy/~/non_spatial_conv/~/conv2_d')
        grads.pop('non_spatial_conv_policy/~/non_spatial_conv_1/~/conv2_d')
        self.assert_tree_leaves_all_zero(grads)

    @flagsaver.flagsaver(**_small_3x3_linear_model_flags(),
                         qcomplete_temp=1e6,
                         add_value_loss=False,
                         add_hypo_value_loss=False,
                         add_decode_loss=False,
                         add_hypo_decode_loss=False,
                         add_policy_loss=True,
                         dtype='float32')
    def test_policy_loss_with_high_temperature_returns_zero_gradients(self):
        all_models_build_config = models.get_all_models_build_config(
            FLAGS.board_size, FLAGS.dtype)
        go_model, params = models.build_model_with_params(
            all_models_build_config, jax.random.PRNGKey(FLAGS.rng))
        params = jax.tree_util.tree_map(
            lambda x: jax.random.normal(
                jax.random.PRNGKey(42), x.shape, dtype=FLAGS.dtype), params)
        game_data = _ones_like_game_data(FLAGS.board_size,
                                         FLAGS.batch_size,
                                         hypo_steps=1)

        grads: dict
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        grads, _ = losses.compute_loss_gradients_and_metrics(
            go_model, params, game_data, rng_key)

        self.assert_tree_leaves_all_close_to_zero(
            grads['non_spatial_conv_embed/~/non_spatial_conv/~/conv2_d'],
            atol=1e-5)
        self.assert_tree_leaves_all_close_to_zero(
            grads['non_spatial_conv_policy/~/non_spatial_conv/~/conv2_d'],
            atol=1e-5)
        self.assert_tree_leaves_all_close_to_zero(
            grads['non_spatial_conv_policy/~/non_spatial_conv_1/~/conv2_d'],
            atol=1e-5)

    @flagsaver.flagsaver(**_small_3x3_linear_model_flags(),
                         qcomplete_temp=0.1,
                         add_value_loss=False,
                         add_decode_loss=False,
                         add_policy_loss=True)
    def test_policy_loss_with_low_temperature_returns_nonzero_gradients(self):
        all_models_build_config = models.get_all_models_build_config(
            FLAGS.board_size, FLAGS.dtype)
        go_model, params = models.build_model_with_params(
            all_models_build_config, jax.random.PRNGKey(FLAGS.rng))
        params = jax.tree_util.tree_map(
            lambda x: jax.random.normal(
                jax.random.PRNGKey(42), x.shape, dtype=FLAGS.dtype), params)
        game_data = _ones_like_game_data(FLAGS.board_size,
                                         FLAGS.batch_size,
                                         hypo_steps=1)

        grads: dict
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        grads, _ = losses.compute_loss_gradients_and_metrics(
            go_model, params, game_data, rng_key)
        self.assert_tree_leaves_any_non_zero(
            grads['non_spatial_conv_embed/~/non_spatial_conv/~/conv2_d'])
        self.assert_tree_leaves_any_non_zero(
            grads['non_spatial_conv_policy/~/non_spatial_conv/~/conv2_d'])
        self.assert_tree_leaves_any_non_zero(
            grads['non_spatial_conv_policy/~/non_spatial_conv_1/~/conv2_d'])

    @flagsaver.flagsaver(**_small_3x3_linear_model_flags(),
                         add_value_loss=True,
                         add_decode_loss=False,
                         add_policy_loss=False)
    def test_value_loss_only_affects_embed_transition_and_value_gradients(
            self):
        all_models_build_config = models.get_all_models_build_config(
            FLAGS.board_size, FLAGS.dtype)
        go_model, params = models.build_model_with_params(
            all_models_build_config, jax.random.PRNGKey(FLAGS.rng))
        params = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), params)
        game_data = _ones_like_game_data(FLAGS.board_size,
                                         FLAGS.batch_size,
                                         hypo_steps=1)

        grads: dict
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        grads, _ = losses.compute_loss_gradients_and_metrics(
            go_model, params, game_data, rng_key)

        self.assert_tree_leaves_all_non_zero(
            grads['non_spatial_conv_embed/~/non_spatial_conv/~/conv2_d'])
        self.assert_tree_leaves_all_non_zero(
            grads['non_spatial_conv_value/~/non_spatial_conv/~/conv2_d'])
        self.assert_tree_leaves_all_non_zero(
            grads['non_spatial_conv_transition/~/non_spatial_conv/~/conv2_d'])
        # Check the remaining gradients are zero.
        grads.pop('non_spatial_conv_embed/~/non_spatial_conv/~/conv2_d')
        grads.pop('non_spatial_conv_value/~/non_spatial_conv/~/conv2_d')
        grads.pop('non_spatial_conv_transition/~/non_spatial_conv/~/conv2_d')
        self.assert_tree_leaves_all_zero(grads)

    @flagsaver.flagsaver(**_small_3x3_linear_model_flags(),
                         add_value_loss=False,
                         add_hypo_value_loss=False,
                         add_decode_loss=True,
                         add_hypo_decode_loss=True,
                         add_policy_loss=False)
    def test_decode_loss_only_affects_embed_transition_and_decode_gradients(
            self):
        all_models_build_config = models.get_all_models_build_config(
            FLAGS.board_size, FLAGS.dtype)
        go_model, params = models.build_model_with_params(
            all_models_build_config, jax.random.PRNGKey(FLAGS.rng))
        params = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), params)
        params[
            'non_spatial_conv_decode/~/non_spatial_conv/~/conv2_d'] = jax.tree_util.tree_map(
                lambda x: -1 * jnp.ones_like(x),
                params['non_spatial_conv_decode/~/non_spatial_conv/~/conv2_d'])
        game_data = _ones_like_game_data(FLAGS.board_size,
                                         FLAGS.batch_size,
                                         hypo_steps=1)

        grads: dict
        rng_key = jax.random.PRNGKey(FLAGS.rng)
        grads, _ = losses.compute_loss_gradients_and_metrics(
            go_model, params, game_data, rng_key)

        self.assert_tree_leaves_all_non_zero(
            grads['non_spatial_conv_embed/~/non_spatial_conv/~/conv2_d'])
        self.assert_tree_leaves_all_non_zero(
            grads['non_spatial_conv_transition/~/non_spatial_conv/~/conv2_d'])
        self.assert_tree_leaves_all_non_zero(
            grads['non_spatial_conv_decode/~/non_spatial_conv/~/conv2_d'])
        # Check the remaining gradients are zero.
        grads.pop('non_spatial_conv_embed/~/non_spatial_conv/~/conv2_d')
        grads.pop('non_spatial_conv_transition/~/non_spatial_conv/~/conv2_d')
        grads.pop('non_spatial_conv_decode/~/non_spatial_conv/~/conv2_d')
        self.assert_tree_leaves_all_zero(grads)

    @flagsaver.flagsaver(board_size=5,
                         embed_model='IdentityEmbed',
                         value_model='TrompTaylorValue',
                         policy_model='Linear3DPolicy',
                         transition_model='RealTransition',
                         loss_sample_action_size=26)
    def test_policy_bias_learns_good_move_from_tromp_taylor(self):
        states = gojax.decode_states("""
                                    B W _ _ _
                                    B W _ _ 
                                    _ _ _ _ W
                                    _ _ W B B
                                    _ _ W B B
                                    TURN=W
                                    """)
        game_data = losses.GameData(nk_states=jnp.expand_dims(states, axis=0),
                                    nk_actions=jnp.full((1, 1),
                                                        fill_value=-1,
                                                        dtype='uint16'),
                                    nk_player_labels=jnp.full((1, 1),
                                                              fill_value=-1,
                                                              dtype='uint16'))
        rng_key = jax.random.PRNGKey(42)
        all_models_build_config = models.get_all_models_build_config(
            FLAGS.board_size, FLAGS.dtype)
        go_model, params = models.build_model_with_params(
            all_models_build_config, rng_key)
        params = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
        grads: optax.Params
        grads, _ = losses.compute_loss_gradients_and_metrics(
            go_model, params, game_data, rng_key)
        bias_grads = grads['linear3_d_policy']['action_b'].flatten()
        _, top_2_moves = jax.lax.top_k(-bias_grads, k=2)
        # The top two moves should be to capture the bottom right black group or
        # the top left black group.
        np.testing.assert_array_equal(top_2_moves, [13, 10])


if __name__ == '__main__':
    absltest.main()
