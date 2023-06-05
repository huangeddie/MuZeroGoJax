"""Defines the training graph."""
import functools
from typing import Callable, Optional

import chex
import haiku as hk
import jax
import jax.nn
import jax.numpy as jnp
import jax.random
import optax
from absl import flags
from jax import lax

from muzero_gojax import data, game, logger, losses, models

_MODEL_UPDATES_PER_TRAIN_STEP = flags.DEFINE_integer(
    'model_updates_per_train_step', 1,
    'Number of model updates per train step to run.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 2,
                                   'Size of the batch to train_model on.')
_TRAJECTORY_LENGTH = flags.DEFINE_integer(
    'trajectory_length', 26, 'Maximum number of game steps for Go.'
    'Usually set to 2(board_size^2).')
_MAX_HYPOTHETICAL_STEPS = flags.DEFINE_integer(
    'max_hypothetical_steps', 1,
    'Maximum number of hypothetical steps to take during training. The number '
    'of hypothetical steps is sampled uniformly from '
    '[1, max_hypothetical_steps].')
_SELF_PLAY_SAMPLE_ACTION_SIZE = flags.DEFINE_integer(
    'self_play_sample_action_size', 0,
    'Number of actions to sample for policy improvement during self play.')
_SELF_PLAY_MODEL = flags.DEFINE_string(
    'self_play_model', None, 'Which model to use to generate trajectories. '
    'Defaults to using the model in training.')


@chex.dataclass(frozen=True)
class StepData:
    """Train step data."""
    trajectory_buffer: data.TrajectoryBuffer  # Sharded
    game_stats: game.GameStats  # Sharded
    params: optax.Params  # Replicated
    opt_state: optax.OptState  # Replicated
    loss_metrics: losses.LossMetrics  # Sharded
    rng_key: jax.random.KeyArray  # Sharded


@chex.dataclass(frozen=True)
class TrainData:
    """Constant data about the training process."""
    board_size: int
    pmap: bool
    trajectory_buffer_size: int
    # TODO: Add batch size.


def _get_self_play_policy_model() -> models.PolicyModel:
    if _SELF_PLAY_MODEL.value == 'random':
        logger.log("Setting initial self play model as random.")
        policy_model = models.get_policy_model(
            models.make_random_policy_tromp_taylor_value_model(), params={})
    elif _SELF_PLAY_MODEL.value == 'tromp_taylor':
        logger.log("Setting initial self play model as Tromp Taylor.")
        policy_model = models.get_policy_model(
            models.make_tromp_taylor_model(), params={})
    elif _SELF_PLAY_MODEL.value == 'tromp_taylor_amplified':
        logger.log(
            "Setting initial self play model as Tromp Taylor Amplified.")
        policy_model = models.get_policy_model(
            models.make_tromp_taylor_amplified_model(), params={})
    elif _SELF_PLAY_MODEL.value is not None and _SELF_PLAY_MODEL.value != '':
        # Load the specified model for self-play game generation.
        logger.log(
            f"Loading initial self play model from {_SELF_PLAY_MODEL.value}")
        self_play_model_transform, self_play_model_params, _ = models.load_model(
            _SELF_PLAY_MODEL.value)
        policy_model = models.get_policy_model(
            self_play_model_transform, self_play_model_params,
            _SELF_PLAY_SAMPLE_ACTION_SIZE.value)
    else:
        logger.log("Self play model will be itself (None).")
        policy_model = None
    return policy_model


def _update_model(go_model: hk.MultiTransformed,
                  optimizer: optax.GradientTransformation, pmap: bool, _: int,
                  step_data: StepData) -> StepData:
    """Updates the model parameters based on the existing trajectories.

    Args:
        go_model (hk.MultiTransformed): Go model.
        optimizer (optax.GradientTransformation): Optimizer.
        step_data (StepData): Training data.

    Returns:
        StepData: Training data with updated parmaeters, optimizer state, 
        RNG key and loss metrics.
    """
    logger.log('Tracing update step')
    rng_key, subkey = jax.random.split(step_data.rng_key)
    logger.log('Tracing sample game data')
    game_data: data.GameData = data.sample_game_data(
        step_data.trajectory_buffer, subkey, _MAX_HYPOTHETICAL_STEPS.value)
    del subkey
    rng_key, subkey = jax.random.split(rng_key)
    logger.log('Tracing compute loss gradients and metrics')
    grads, loss_metrics = losses.compute_loss_gradients_and_metrics(
        go_model, step_data.params, game_data, subkey)
    if pmap:
        grads = jax.lax.pmean(grads, axis_name='num_devices')
        loss_metrics = jax.lax.pmean(loss_metrics, axis_name='num_devices')
    del subkey
    logger.log('Tracing update model')
    model_updates, opt_state = optimizer.update(grads, step_data.opt_state,
                                                step_data.params)
    params = optax.apply_updates(step_data.params, model_updates)
    return step_data.replace(params=params,
                             opt_state=opt_state,
                             rng_key=rng_key,
                             loss_metrics=jax.tree_map(
                                 lambda x: x.astype(jnp.float32),
                                 loss_metrics))


def _step(board_size: int, self_play_policy: Optional[models.PolicyModel],
          go_model: hk.MultiTransformed,
          optimizer: optax.GradientTransformation, pmap: bool, train_step: int,
          step_data: StepData) -> StepData:
    """
    Executes a single train step comprising self-play, and a model update.
    :param board_size: board size.
    :param self_play_policy: Policy to generate games.
    :param go_model: JAX-Haiku model architecture.
    :param train_step: Train step index.
    :param optimizer: Optax optimizer.
    :param step_data: Train data.
    :return:
    """
    logger.log('Tracing train step...')
    rng_key, subkey = jax.random.split(step_data.rng_key)
    if self_play_policy is None:
        logger.log('Tracing self-play policy model.')
        self_play_policy = models.get_policy_model(
            go_model, step_data.params, _SELF_PLAY_SAMPLE_ACTION_SIZE.value)
    logger.log('Tracing self-play.')
    trajectories = game.self_play(
        game.new_trajectories(
            board_size, _BATCH_SIZE.value //
            jax.local_device_count() if pmap else _BATCH_SIZE.value,
            _TRAJECTORY_LENGTH.value), self_play_policy, subkey)
    trajectory_buffer = data.mod_insert_trajectory(step_data.trajectory_buffer,
                                                   trajectories, train_step)
    del subkey
    logger.log('Tracing game stats.')
    game_stats = game.get_game_stats(trajectories)
    if pmap:
        game_stats = jax.lax.pmean(game_stats, axis_name='num_devices')
    logger.log('Tracing trajectory augmentation.')
    _, subkey = jax.random.split(rng_key)
    updated_step_data = jax.lax.fori_loop(
        0, _MODEL_UPDATES_PER_TRAIN_STEP.value,
        jax.tree_util.Partial(_update_model, go_model, optimizer, pmap),
        step_data.replace(trajectory_buffer=trajectory_buffer,
                          game_stats=game_stats,
                          rng_key=subkey))
    chex.assert_trees_all_equal_shapes(updated_step_data, step_data)
    chex.assert_trees_all_equal_dtypes(updated_step_data, step_data)
    logger.log('Tracing train step done.')
    return updated_step_data


def _init_loss_metrics() -> losses.LossMetrics:
    """Initializes the train metrics with zeros"""
    return losses.LossMetrics(
        area_loss=jnp.zeros(()),
        area_acc=jnp.zeros(()),
        value_loss=jnp.zeros(()),
        value_acc=jnp.zeros(()),
        policy_loss=jnp.zeros(()),
        policy_acc=jnp.zeros(()),
        policy_entropy=jnp.zeros(()),
        partial_qval_entropy=jnp.zeros(()),
        hypo_area_loss=jnp.zeros(()),
        hypo_area_acc=jnp.zeros(()),
        hypo_value_loss=jnp.zeros(()),
        hypo_value_acc=jnp.zeros(()),
    )


def _init_game_stats() -> game.GameStats:
    """Initializes the game stats with zeros."""
    return game.GameStats(avg_game_length=jnp.zeros(()),
                          black_win_pct=jnp.zeros(()),
                          tie_pct=jnp.zeros(()),
                          white_win_pct=jnp.zeros(()),
                          piece_collision_rate=jnp.zeros(()),
                          pass_rate=jnp.zeros(()))


def init_step_data(board_size: int, trajectory_buffer_size: int,
                   params: optax.Params, opt_state: optax.OptState,
                   rng_key: jax.random.KeyArray) -> StepData:
    """Initializes the training data."""
    return StepData(trajectory_buffer=data.init_trajectory_buffer(
        trajectory_buffer_size, _BATCH_SIZE.value, _TRAJECTORY_LENGTH.value,
        board_size),
                    params=params,
                    opt_state=opt_state,
                    loss_metrics=_init_loss_metrics(),
                    rng_key=rng_key,
                    game_stats=_init_game_stats())


def _step_multiple(train_data: TrainData,
                   self_play_policy: Optional[models.PolicyModel],
                   go_model: hk.MultiTransformed,
                   optimizer: optax.GradientTransformation, num_steps: int,
                   step_data: StepData) -> StepData:
    """Executes multiple training steps."""
    # num_steps is marked as a static argument so we can switch between for
    # loops and train steps.
    if num_steps > 1:
        simplified_train_step_fn = jax.tree_util.Partial(
            _step, train_data.board_size, self_play_policy, go_model,
            optimizer, train_data.pmap)
        return lax.fori_loop(0,
                             num_steps,
                             simplified_train_step_fn,
                             init_val=step_data)
    return _step(train_data.board_size, self_play_policy, go_model, optimizer,
                 train_data.pmap, 0, step_data)


def get_multi_step_fn(train_data: TrainData, go_model: hk.MultiTransformed,
                      optimizer: optax.GradientTransformation,
                      num_steps: int) -> Callable[[StepData], StepData]:
    """Returns the multi train step function."""
    self_play_policy = _get_self_play_policy_model()
    if train_data.pmap:
        return jax.pmap(functools.partial(_step_multiple, train_data,
                                          self_play_policy, go_model,
                                          optimizer, num_steps),
                        axis_name='num_devices',
                        donate_argnums=0)
    return jax.jit(functools.partial(_step_multiple, train_data,
                                     self_play_policy, go_model, optimizer,
                                     num_steps),
                   donate_argnums=0)
