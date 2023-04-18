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
from muzero_gojax import data, losses, main, models, nt_utils

FLAGS = main.FLAGS


def _ones_like_game_data(board_size: int, batch_size: int,
                         hypo_steps: int) -> data.GameData:
    nk_states = nt_utils.unflatten_first_dim(
        jnp.ones_like(
            gojax.new_states(board_size, batch_size * (hypo_steps + 1))),
        batch_size, (hypo_steps + 1))
    nk_player_labels = -jnp.ones(
        (batch_size, 2, board_size, board_size), dtype='int8')
    return data.GameData(start_states=nk_states[:, 0],
                         end_states=nk_states[:, 1],
                         nk_actions=jnp.ones((batch_size, (hypo_steps + 1)),
                                             dtype='uint16'),
                         start_player_final_areas=nk_player_labels,
                         end_player_final_areas=nk_player_labels)


def _small_3x3_linear_model_flags():
    return {
        'board_size': 3,
        'hdim': 2,
        'embed_nlayers': 0,
        'value_nlayers': 0,
        'area_nlayers': 0,
        'policy_nlayers': 0,
        'transition_nlayers': 0,
        'embed_model': 'NonSpatialConvEmbed',
        'value_model': 'NonSpatialConvValue',
        'area_model': 'LinearConvArea',
        'policy_model': 'NonSpatialConvPolicy',
        'transition_model': 'NonSpatialConvTransition'
    }


class ComputeLossGradientsAndMetricsTestCase(chex.TestCase):
    """Test losses.py"""

    def assert_tree_leaves_all_zero(self, pytree: chex.ArrayTree):
        chex.assert_trees_all_equal(
            pytree, jax.tree_map(lambda x: jnp.zeros_like(x), pytree))

    def assert_tree_leaves_any_non_zero(self, pytree: chex.ArrayTree):
        if not jax.tree_util.tree_reduce(
                lambda any_non_zero, leaf: any_non_zero or jnp.any(leaf),
                pytree,
                initializer=False):
            self.fail("")

    def assert_tree_leaves_all_non_zero(self, pytree: chex.ArrayTree):
        binary_pytree = jax.tree_map(lambda x: x.astype(bool), pytree)
        chex.assert_trees_all_equal(
            binary_pytree,
            jax.tree_map(lambda x: jnp.ones_like(x), binary_pytree))

    def setUp(self):
        FLAGS.mark_as_parsed()

    @flagsaver.flagsaver(**_small_3x3_linear_model_flags(),
                         add_value_loss=False,
                         add_hypo_value_loss=False,
                         add_area_loss=False,
                         add_hypo_area_loss=False,
                         add_policy_loss=False)
    def test_no_loss_returns_no_gradients(self):
        all_models_build_config = models.get_all_models_build_config(
            FLAGS.board_size, FLAGS.dtype)
        go_model, params = models.build_model_with_params(
            all_models_build_config, jax.random.PRNGKey(FLAGS.rng))
        params = jax.tree_map(lambda x: jnp.ones_like(x), params)
        game_data = _ones_like_game_data(FLAGS.board_size,
                                         FLAGS.batch_size,
                                         hypo_steps=1)

        rng_key = jax.random.PRNGKey(FLAGS.rng)
        grads, _ = losses.compute_loss_gradients_and_metrics(
            go_model, params, game_data, rng_key)
        self.assert_tree_leaves_all_zero(grads)

    @flagsaver.flagsaver(**_small_3x3_linear_model_flags(),
                         add_value_loss=True,
                         add_area_loss=True,
                         add_policy_loss=True)
    def test_all_loss_gradients_are_finite(self):
        all_models_build_config = models.get_all_models_build_config(
            FLAGS.board_size, FLAGS.dtype)
        go_model, params = models.build_model_with_params(
            all_models_build_config, jax.random.PRNGKey(FLAGS.rng))
        params = jax.tree_map(lambda x: jnp.ones_like(x), params)
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
                         add_area_loss=False,
                         add_hypo_area_loss=False,
                         add_policy_loss=True)
    def test_policy_loss_only_affects_embed_and_policy_gradients(self):
        all_models_build_config = models.get_all_models_build_config(
            FLAGS.board_size, FLAGS.dtype)
        go_model, params = models.build_model_with_params(
            all_models_build_config, jax.random.PRNGKey(FLAGS.rng))
        params = jax.tree_map(
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
                         add_value_loss=True,
                         add_area_loss=False,
                         add_policy_loss=False)
    def test_value_loss_only_affects_embed_transition_and_value_gradients(
            self):
        all_models_build_config = models.get_all_models_build_config(
            FLAGS.board_size, FLAGS.dtype)
        go_model, params = models.build_model_with_params(
            all_models_build_config, jax.random.PRNGKey(FLAGS.rng))
        params = jax.tree_map(lambda x: jnp.ones_like(x), params)
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
                         add_area_loss=True,
                         add_hypo_area_loss=True,
                         add_policy_loss=False)
    def test_area_loss_only_affects_embed_transition_and_area_gradients(self):
        all_models_build_config = models.get_all_models_build_config(
            FLAGS.board_size, FLAGS.dtype)
        go_model, params = models.build_model_with_params(
            all_models_build_config, jax.random.PRNGKey(FLAGS.rng))
        params = jax.tree_map(lambda x: jnp.ones_like(x), params)
        params['linear_conv_area/~/conv2_d'] = jax.tree_map(
            lambda x: -1 * jnp.ones_like(x),
            params['linear_conv_area/~/conv2_d'])
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
            grads['linear_conv_area/~/conv2_d'])
        # Check the remaining gradients are zero.
        grads.pop('non_spatial_conv_embed/~/non_spatial_conv/~/conv2_d')
        grads.pop('non_spatial_conv_transition/~/non_spatial_conv/~/conv2_d')
        grads.pop('linear_conv_area/~/conv2_d')
        self.assert_tree_leaves_all_zero(grads)

    @flagsaver.flagsaver(board_size=5,
                         embed_model='IdentityEmbed',
                         value_model='TrompTaylorValue',
                         policy_model='Linear3DPolicy',
                         transition_model='RealTransition',
                         loss_sample_action_size=26,
                         qval_scale=1)
    def test_policy_bias_learns_good_move_from_tromp_taylor(self):
        states = gojax.decode_states("""
                                    B W _ _ _
                                    B W _ _ 
                                    _ _ _ _ W
                                    _ _ W B B
                                    _ _ W B B
                                    TURN=W
                                    """)
        player_final_areas = jnp.full((1, 2, 5, 5),
                                      fill_value=-1,
                                      dtype='uint16')
        game_data = data.GameData(start_states=states,
                                  end_states=states,
                                  nk_actions=jnp.full((1, 1),
                                                      fill_value=-1,
                                                      dtype='uint16'),
                                  start_player_final_areas=player_final_areas,
                                  end_player_final_areas=player_final_areas)
        rng_key = jax.random.PRNGKey(42)
        all_models_build_config = models.get_all_models_build_config(
            FLAGS.board_size, FLAGS.dtype)
        go_model, params = models.build_model_with_params(
            all_models_build_config, rng_key)
        params = jax.tree_map(lambda x: jnp.zeros_like(x), params)
        grads: optax.Params
        grads, _ = losses.compute_loss_gradients_and_metrics(
            go_model, params, game_data, rng_key)
        bias_grads = grads['linear3_d_policy']['action_b'].flatten()
        _, top_2_moves = jax.lax.top_k(-bias_grads, k=2)
        # The top two moves should be to capture the bottom right black group or
        # the top left black group.
        np.testing.assert_array_equal(top_2_moves, [13, 10], str(bias_grads))


if __name__ == '__main__':
    absltest.main()
