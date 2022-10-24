"""Tests losses.py."""
# pylint: disable=missing-function-docstring,no-value-for-parameter,too-many-public-methods,duplicate-code
import chex
import gojax
import jax.random
from absl.testing import absltest
from absl.testing import flagsaver
from jax import numpy as jnp

from muzero_gojax import game
from muzero_gojax import losses
from muzero_gojax import main
from muzero_gojax import models
from muzero_gojax import nt_utils

FLAGS = main.FLAGS


def _ones_like_trajectories(board_size: int, batch_size: int,
                            trajectory_length: int) -> game.Trajectories:
    nt_states = nt_utils.unflatten_first_dim(
        jnp.ones_like(gojax.new_states(board_size, batch_size * trajectory_length)), batch_size,
        trajectory_length)
    return game.Trajectories(nt_states=nt_states,
                             nt_actions=jnp.ones((batch_size, trajectory_length), dtype='uint16'))


def _small_3x3_linear_model_flags():
    return {
        'board_size': 3, 'hdim': 2, 'embed_model': 'linear_conv', 'value_model': 'linear_conv',
        'decode_model': 'linear_conv', 'policy_model': 'linear_conv',
        'transition_model': 'linear_conv'
    }


class ComputeLossGradientsAndMetricsTestCase(chex.TestCase):
    """Test losses.py"""

    def assert_tree_leaves_all_zero(self, pytree: chex.ArrayTree):
        chex.assert_trees_all_equal(pytree,
                                    jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), pytree))

    def assert_tree_leaves_any_non_zero(self, pytree: chex.ArrayTree):
        if not jax.tree_util.tree_reduce(lambda any_non_zero, leaf: any_non_zero or jnp.any(leaf),
                                         pytree, initializer=False):
            self.fail("")

    def assert_tree_leaves_all_non_zero(self, pytree: chex.ArrayTree):
        binary_pytree = jax.tree_util.tree_map(lambda x: x.astype(bool), pytree)
        chex.assert_trees_all_equal(binary_pytree,
                                    jax.tree_util.tree_map(lambda x: jnp.ones_like(x),
                                                           binary_pytree))

    def assert_tree_leaves_all_close_to_zero(self, pytree: chex.ArrayTree, atol: float):
        chex.assert_trees_all_close(pytree,
                                    jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), pytree),
                                    atol=atol)

    def test_assert_tree_leaves_all_zero(self):
        self.assert_tree_leaves_all_zero({'a': jnp.zeros(()), 'b': {'c': jnp.zeros(2)}})
        with self.assertRaises(AssertionError):
            self.assert_tree_leaves_all_zero({'a': jnp.zeros(()), 'b': {'c': jnp.array([0, 1])}})

    def test_assert_tree_leaves_any_non_zero(self):
        self.assert_tree_leaves_any_non_zero({'a': jnp.zeros(()), 'b': {'c': jnp.array([0, 1])}})
        with self.assertRaises(AssertionError):
            self.assert_tree_leaves_any_non_zero({'a': jnp.zeros(()), 'b': {'c': jnp.zeros(2)}})

    def test_assert_tree_leaves_all_non_zero(self):
        self.assert_tree_leaves_all_non_zero({'a': jnp.ones(()), 'b': {'c': jnp.ones(2)}})
        with self.assertRaises(AssertionError):
            self.assert_tree_leaves_all_non_zero({'a': jnp.ones(()), 'b': {'c': jnp.array([0, 1])}})

    def test_assert_pytree_allclose(self):
        self.assert_tree_leaves_all_close_to_zero(
            {'a': jnp.array(1e-7), 'b': {'c': jnp.array(1e-5)}}, atol=1e-4)
        with self.assertRaises(AssertionError):
            self.assert_tree_leaves_all_close_to_zero(
                {'a': jnp.array(0.5), 'b': {'c': jnp.array(1e-5)}}, atol=1e-4)

    def setUp(self):
        FLAGS.mark_as_parsed()

    @flagsaver.flagsaver(**_small_3x3_linear_model_flags(), add_value_loss=False,
                         add_decode_loss=False, add_policy_loss=False, add_trans_loss=False)
    def test_no_loss_returns_no_gradients(self):
        go_model, params = models.make_model(FLAGS.board_size)
        params = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), params)
        trajectories = _ones_like_trajectories(FLAGS.board_size, FLAGS.batch_size,
                                               FLAGS.trajectory_length)

        grads, _ = losses.compute_loss_gradients_and_metrics(go_model, params, trajectories)
        self.assert_tree_leaves_all_zero(grads)

    @flagsaver.flagsaver(**_small_3x3_linear_model_flags(), add_value_loss=False,
                         add_decode_loss=False, add_policy_loss=False, add_trans_loss=True)
    def test_transition_loss_only_affects_transition_gradients(self):
        go_model, params = models.make_model(FLAGS.board_size)
        params = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), params)
        trajectories = _ones_like_trajectories(FLAGS.board_size, FLAGS.batch_size,
                                               FLAGS.trajectory_length)

        grads: dict
        grads, _ = losses.compute_loss_gradients_and_metrics(go_model, params, trajectories)

        self.assert_tree_leaves_any_non_zero(grads['linear_conv_transition/~/conv2_d'])
        with self.assertRaises(AssertionError):
            self.assert_tree_leaves_all_non_zero(grads['linear_conv_transition/~/conv2_d'])
        # Check the remaining gradients are zero.
        grads.pop('linear_conv_transition/~/conv2_d')
        self.assert_tree_leaves_all_zero(grads)

    @flagsaver.flagsaver(**_small_3x3_linear_model_flags(), add_value_loss=False,
                         add_decode_loss=False, add_policy_loss=True, add_trans_loss=False)
    def test_policy_loss_only_affects_embed_and_policy_gradients(self):
        go_model, params = models.make_model(FLAGS.board_size)
        params = jax.tree_util.tree_map(
            lambda x: jax.random.normal(jax.random.PRNGKey(42), x.shape, dtype='bfloat16'), params)
        trajectories = _ones_like_trajectories(FLAGS.board_size, FLAGS.batch_size,
                                               FLAGS.trajectory_length)

        grads: dict
        grads, _ = losses.compute_loss_gradients_and_metrics(go_model, params, trajectories)

        self.assert_tree_leaves_all_non_zero(grads['linear_conv_embed/~/conv2_d'])
        self.assert_tree_leaves_all_non_zero(grads['linear_conv_policy/~/conv2_d'])
        self.assert_tree_leaves_all_non_zero(grads['linear_conv_policy/~/conv2_d_1'])
        # Check the remaining gradients are zero.
        grads.pop('linear_conv_embed/~/conv2_d')
        grads.pop('linear_conv_policy/~/conv2_d')
        grads.pop('linear_conv_policy/~/conv2_d_1')
        self.assert_tree_leaves_all_zero(grads)

    @flagsaver.flagsaver(**_small_3x3_linear_model_flags(), add_value_loss=False,
                         add_decode_loss=False, add_policy_loss=True, add_trans_loss=False)
    def test_constant_trans_values_makes_policy_loss_optimal(self):
        go_model, params = models.make_model(FLAGS.board_size)
        params = jax.tree_util.tree_map(
            lambda x: jax.random.normal(jax.random.PRNGKey(42), x.shape, dtype='bfloat16'), params)
        zero_value_params = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x),
                                                   params['linear_conv_value/~/conv2_d'])
        params['linear_conv_value/~/conv2_d'] = zero_value_params
        trajectories = _ones_like_trajectories(FLAGS.board_size, FLAGS.batch_size,
                                               FLAGS.trajectory_length)

        grads: dict
        grads, _ = losses.compute_loss_gradients_and_metrics(go_model, params, trajectories)

        self.assert_tree_leaves_all_zero(grads['linear_conv_embed/~/conv2_d'])
        self.assert_tree_leaves_all_zero(grads['linear_conv_policy/~/conv2_d'])
        self.assert_tree_leaves_all_zero(grads['linear_conv_policy/~/conv2_d_1'])

    @flagsaver.flagsaver(**_small_3x3_linear_model_flags(), temperature=100, add_value_loss=False,
                         add_decode_loss=False, add_policy_loss=True, add_trans_loss=False)
    def test_policy_loss_with_high_temperature_returns_zero_gradients(self):
        go_model, params = models.make_model(FLAGS.board_size)
        params = jax.tree_util.tree_map(
            lambda x: jax.random.normal(jax.random.PRNGKey(42), x.shape, dtype='bfloat16'), params)
        trajectories = _ones_like_trajectories(FLAGS.board_size, FLAGS.batch_size,
                                               FLAGS.trajectory_length)

        grads: dict
        grads, _ = losses.compute_loss_gradients_and_metrics(go_model, params, trajectories)

        self.assert_tree_leaves_all_close_to_zero(grads['linear_conv_embed/~/conv2_d'], atol=1e-6)
        self.assert_tree_leaves_all_close_to_zero(grads['linear_conv_policy/~/conv2_d'], atol=1e-6)
        self.assert_tree_leaves_all_close_to_zero(grads['linear_conv_policy/~/conv2_d_1'],
                                                  atol=1e-6)

    @flagsaver.flagsaver(**_small_3x3_linear_model_flags(), temperature=0.01, add_value_loss=False,
                         add_decode_loss=False, add_policy_loss=True, add_trans_loss=False)
    def test_policy_loss_with_low_temperature_returns_nonzero_gradients(self):
        go_model, params = models.make_model(FLAGS.board_size)
        params = jax.tree_util.tree_map(
            lambda x: jax.random.normal(jax.random.PRNGKey(42), x.shape, dtype='bfloat16'), params)
        trajectories = _ones_like_trajectories(FLAGS.board_size, FLAGS.batch_size,
                                               FLAGS.trajectory_length)

        grads: dict
        grads, _ = losses.compute_loss_gradients_and_metrics(go_model, params, trajectories)

        self.assert_tree_leaves_all_non_zero(grads['linear_conv_embed/~/conv2_d'])
        self.assert_tree_leaves_all_non_zero(grads['linear_conv_policy/~/conv2_d'])
        self.assert_tree_leaves_all_non_zero(grads['linear_conv_policy/~/conv2_d_1'])

    @flagsaver.flagsaver(**_small_3x3_linear_model_flags(), add_value_loss=True,
                         add_decode_loss=False, add_policy_loss=False, add_trans_loss=False)
    def test_value_loss_only_affects_embed_and_value_gradients(self):
        go_model, params = models.make_model(FLAGS.board_size)
        params = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), params)
        trajectories = _ones_like_trajectories(FLAGS.board_size, FLAGS.batch_size,
                                               FLAGS.trajectory_length)

        grads: dict
        grads, _ = losses.compute_loss_gradients_and_metrics(go_model, params, trajectories)

        self.assert_tree_leaves_all_non_zero(grads['linear_conv_embed/~/conv2_d'])
        self.assert_tree_leaves_all_non_zero(grads['linear_conv_value/~/conv2_d'])
        # Check the remaining gradients are zero.
        grads.pop('linear_conv_embed/~/conv2_d')
        grads.pop('linear_conv_value/~/conv2_d')
        self.assert_tree_leaves_all_zero(grads)

    @flagsaver.flagsaver(**_small_3x3_linear_model_flags(), add_value_loss=False,
                         add_decode_loss=True, add_policy_loss=False, add_trans_loss=False)
    def test_decode_loss_only_affects_embed_and_decode_gradients(self):
        go_model, params = models.make_model(FLAGS.board_size)
        params = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), params)
        params['linear_conv_decode/~/conv2_d'] = jax.tree_util.tree_map(
            lambda x: -1 * jnp.ones_like(x), params['linear_conv_decode/~/conv2_d'])
        trajectories = _ones_like_trajectories(FLAGS.board_size, FLAGS.batch_size,
                                               FLAGS.trajectory_length)

        grads: dict
        grads, _ = losses.compute_loss_gradients_and_metrics(go_model, params, trajectories)

        self.assert_tree_leaves_all_non_zero(grads['linear_conv_embed/~/conv2_d'])
        self.assert_tree_leaves_all_non_zero(grads['linear_conv_decode/~/conv2_d'])
        # Check the remaining gradients are zero.
        grads.pop('linear_conv_embed/~/conv2_d')
        grads.pop('linear_conv_decode/~/conv2_d')
        self.assert_tree_leaves_all_zero(grads)

    @flagsaver.flagsaver(**_small_3x3_linear_model_flags(), add_value_loss=False,
                         add_decode_loss=False, add_policy_loss=False, add_trans_loss=True)
    def test_trans_loss_only_affects_trans_gradients(self):
        go_model, params = models.make_model(FLAGS.board_size)
        trajectories = _ones_like_trajectories(FLAGS.board_size, FLAGS.batch_size,
                                               FLAGS.trajectory_length)

        grads: dict
        grads, _ = losses.compute_loss_gradients_and_metrics(go_model, params, trajectories)

        self.assert_tree_leaves_any_non_zero(grads['linear_conv_transition/~/conv2_d'])
        # Check everything else is non-zero.
        grads.pop('linear_conv_transition/~/conv2_d')
        self.assert_tree_leaves_all_zero(grads)


if __name__ == '__main__':
    absltest.main()
