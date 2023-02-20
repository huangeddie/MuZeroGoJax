"""Manages the MuZero training of Go models."""
import dataclasses
import functools
from typing import Callable, Optional, Tuple

import chex
import haiku as hk
import jax
import jax.nn
import jax.numpy as jnp
import jax.random
import optax
import pandas as pd
from absl import flags
from jax import lax

from muzero_gojax import data, game, logger, losses, metrics, models

_OPTIMIZER = flags.DEFINE_enum('optimizer', 'sgd', ['sgd', 'adam', 'adamw'],
                               'Optimizer.')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 0.01,
                                    'Learning rate for the optimizer.')
_LR_WARMUP_STEPS = flags.DEFINE_integer(
    'lr_warmup_steps', 1, 'Number of training steps to allocate for warmup.')
_TRAINING_STEPS = flags.DEFINE_integer(
    'training_steps', 10,
    'A train step consists of one self-play, followed by multiple model updates.'
)
_MODEL_UPDATES_PER_TRAIN_STEP = flags.DEFINE_integer(
    'model_updates_per_train_step', 1,
    'Number of model updates per train step to run.')
_LOG_TRAINING_FREQUENCY = flags.DEFINE_integer(
    'log_training_frequency', 1, 'How often to log the training steps. '
    'Steps within the frequency are JIT-ed. '
    'Set this value to <= 0 to deactivate the JIT on the train step')
_LOG_LOSS_VALUES = flags.DEFINE_bool('log_loss_values', False,
                                     'Whether to log loss values.')

_BATCH_SIZE = flags.DEFINE_integer('batch_size', 2,
                                   'Size of the batch to train_model on.')
_TRAJECTORY_LENGTH = flags.DEFINE_integer(
    'trajectory_length', 26, 'Maximum number of game steps for Go.'
    'Usually set to 2(board_size^2).')
_SELF_PLAY_MODEL = flags.DEFINE_string(
    'self_play_model', None, 'Which model to use to generate trajectories. '
    'Defaults to using the model in training.')
_SELF_PLAY_SAMPLE_ACTION_SIZE = flags.DEFINE_integer(
    'self_play_sample_action_size', 0,
    'Number of actions to sample for policy improvement during self play.')
_SELF_PLAY_QVAL_SCALE = flags.DEFINE_float(
    'self_play_qval_scale', 0,
    'The value C in the policy improvement expression: '
    'policy_logits + gumbel + C * qval.')
_UPDATE_SELF_PLAY_POLICY_FREQUENCY = flags.DEFINE_integer(
    'update_self_play_policy_frequency', 1,
    'If the self play model transform is the same, how frequently to update '
    'the self play model params. Otherwise not applicable.')
_EVAL_ELO_FREQUENCY = flags.DEFINE_integer(
    'eval_elo_frequency', 0,
    'How often to evaluate the model against the benchmarks during training.')

_MAX_HYPOTHETICAL_STEPS = flags.DEFINE_integer(
    'max_hypothetical_steps', 1,
    'Maximum number of hypothetical steps to take during training. The number '
    'of hypothetical steps is sampled uniformly from '
    '[1, max_hypothetical_steps].')


@chex.dataclass(frozen=True)
class TrainData:
    """Training data."""
    game_stats: game.GameStats
    params: optax.Params
    opt_state: optax.OptState
    loss_metrics: losses.LossMetrics
    rng_key: jax.random.KeyArray


def _update_model(
        grads: optax.Params, optimizer: optax.GradientTransformation,
        params: optax.Params,
        opt_state: optax.OptState) -> Tuple[optax.Params, optax.OptState]:
    """Updates the model in a single train_model step."""
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


def _get_optimizer() -> optax.GradientTransformation:
    """Gets the JAX optimizer for the corresponding name."""
    schedule = optax.linear_schedule(0, _LEARNING_RATE.value,
                                     _LR_WARMUP_STEPS.value)
    return {
        'adam': optax.adam,
        'sgd': optax.sgd,
        'adamw': optax.adamw
    }[_OPTIMIZER.value](schedule)


def _update_step(go_model, optimizer: optax.GradientTransformation,
                 augmented_trajectories: game.Trajectories, _: int,
                 train_data: TrainData) -> TrainData:
    rng_key, subkey = jax.random.split(train_data.rng_key)
    game_data: data.GameData = data.sample_game_data(
        augmented_trajectories, subkey, _MAX_HYPOTHETICAL_STEPS.value)
    del subkey
    rng_key, subkey = jax.random.split(rng_key)
    grads, loss_metrics = losses.compute_loss_gradients_and_metrics(
        go_model, train_data.params, game_data, subkey)
    del subkey
    params, opt_state = _update_model(grads, optimizer, train_data.params,
                                      train_data.opt_state)
    return train_data.replace(params=params,
                              opt_state=opt_state,
                              rng_key=rng_key,
                              loss_metrics=loss_metrics)


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3))
def _train_step(board_size: int,
                self_play_policy: Optional[models.PolicyModel],
                go_model: hk.MultiTransformed,
                optimizer: optax.GradientTransformation, _: int,
                train_data: TrainData) -> TrainData:
    """
    Executes a single train step comprising self-play, and an update.
    :param board_size: board size.
    :param self_play_policy: Policy to generate games.
    :param go_model: JAX-Haiku model architecture.
    :param _: ignored training step index.
    :param optimizer: Optax optimizer.
    :param train_data: Train data.
    :return:
    """
    rng_key, subkey = jax.random.split(train_data.rng_key)
    if self_play_policy is None:
        self_play_policy = models.get_policy_model(
            go_model, train_data.params, _SELF_PLAY_SAMPLE_ACTION_SIZE.value)
    trajectories = game.self_play(
        game.new_trajectories(board_size, _BATCH_SIZE.value,
                              _TRAJECTORY_LENGTH.value), self_play_policy,
        subkey)
    del subkey
    game_stats = game.get_game_stats(trajectories)
    augmented_trajectories: game.Trajectories = game.rotationally_augment_trajectories(
        trajectories)
    _, subkey = jax.random.split(rng_key)
    return jax.lax.fori_loop(
        0, _MODEL_UPDATES_PER_TRAIN_STEP.value,
        jax.tree_util.Partial(_update_step, go_model, optimizer,
                              augmented_trajectories),
        train_data.replace(game_stats=game_stats, rng_key=subkey))


def _get_initial_self_play_policy_model(
        go_model: hk.MultiTransformed,
        params: optax.Params) -> models.PolicyModel:
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
            self_play_model_transform,
            self_play_model_params,
            _SELF_PLAY_SAMPLE_ACTION_SIZE.value,
            qval_scale=_SELF_PLAY_QVAL_SCALE.value)
    elif _UPDATE_SELF_PLAY_POLICY_FREQUENCY.value > 1:
        # By default, use the model in training to generate self-play games.
        logger.log(
            "Self play model will be set as current version of model in training."
        )
        policy_model = models.get_policy_model(
            go_model,
            params,
            _SELF_PLAY_SAMPLE_ACTION_SIZE.value,
            qval_scale=_SELF_PLAY_QVAL_SCALE.value)
    else:
        logger.log("Self play model will be itself (None).")
        policy_model = None
    return policy_model


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def _multiple_train_steps(board_size: int,
                          self_play_policy: Optional[models.PolicyModel],
                          go_model: hk.MultiTransformed,
                          optimizer: optax.GradientTransformation,
                          num_steps: int, train_data: TrainData) -> TrainData:
    """Executes multiple training steps."""
    # num_steps is marked as a static argument so we can switch between for
    # loops and train steps.
    if num_steps > 1:
        simplified_train_step_fn = jax.tree_util.Partial(
            _train_step, board_size, self_play_policy, go_model, optimizer)
        return lax.fori_loop(0,
                             num_steps,
                             simplified_train_step_fn,
                             init_val=train_data)
    return _train_step(board_size, self_play_policy, go_model, optimizer, 0,
                       train_data)


def _init_loss_metrics(dtype: str) -> losses.LossMetrics:
    """Initializes the train metrics with zeros with the dtype."""
    return losses.LossMetrics(
        decode_loss=jnp.zeros((), dtype=dtype),
        decode_acc=jnp.zeros((), dtype=dtype),
        value_loss=jnp.zeros((), dtype=dtype),
        value_acc=jnp.zeros((), dtype=dtype),
        policy_loss=jnp.zeros((), dtype=dtype),
        policy_acc=jnp.zeros((), dtype=dtype),
        policy_entropy=jnp.zeros((), dtype=dtype),
        hypo_decode_loss=jnp.zeros((), dtype=dtype),
        hypo_decode_acc=jnp.zeros((), dtype=dtype),
        hypo_value_loss=jnp.zeros((), dtype=dtype),
        hypo_value_acc=jnp.zeros((), dtype=dtype),
    )


def _get_train_step_log_data(train_data):
    log_train_step_data = dataclasses.asdict(train_data.loss_metrics)
    if not _LOG_LOSS_VALUES.value:
        log_train_step_data = {
            k: v
            for k, v in log_train_step_data.items() if not k.endswith('loss')
        }
    log_train_step_data.update(dataclasses.asdict(train_data.game_stats))
    return jax.tree_util.tree_map(lambda x: round(x.item(), 3),
                                  log_train_step_data)


def train_model(
        go_model: hk.MultiTransformed, params: optax.Params, board_size: int,
        dtype: str,
        rng_key: jax.random.KeyArray) -> Tuple[optax.Params, pd.DataFrame]:
    """
    Trains the model with the specified hyperparameters.

    :param go_model: JAX-Haiku model architecture.
    :param params: Model parameters.
    :param board_size: Board size.
    :return: The model parameters and a metric log dataframe.
    """
    if _TRAINING_STEPS.value <= 0:
        # Return early.
        return params, pd.json_normalize([])

    optimizer = _get_optimizer()
    opt_state = optimizer.init(params)

    train_data = TrainData(params=params,
                           opt_state=opt_state,
                           loss_metrics=_init_loss_metrics(dtype),
                           rng_key=rng_key,
                           game_stats=game.GameStats())
    self_play_policy = _get_initial_self_play_policy_model(go_model, params)
    metrics_logs = []
    for multi_step in range(
            max(_LOG_TRAINING_FREQUENCY.value, 1),
            _TRAINING_STEPS.value + max(_LOG_TRAINING_FREQUENCY.value, 1),
            max(_LOG_TRAINING_FREQUENCY.value, 1)):
        try:
            train_data = _multiple_train_steps(board_size, self_play_policy,
                                               go_model, optimizer,
                                               _LOG_TRAINING_FREQUENCY.value,
                                               train_data)
        except KeyboardInterrupt:
            logger.log("Caught keyboard interrupt. Ending training early.")
            break
        metrics_logs.append(_get_train_step_log_data(train_data))
        logger.log(f'{multi_step}: {metrics_logs[-1]}')

        if (_UPDATE_SELF_PLAY_POLICY_FREQUENCY.value > 1 and
                multi_step % _UPDATE_SELF_PLAY_POLICY_FREQUENCY.value == 0):
            logger.log(
                "Updating self play policy with deep copy of training model.")
            self_play_policy = models.get_policy_model(
                go_model, jax.tree_util.tree_map(jnp.copy, train_data.params),
                _SELF_PLAY_SAMPLE_ACTION_SIZE.value)
            logger.log("Resetting optimizer state.")
            train_data = train_data.replace(
                opt_state=optimizer.init(train_data.params))

        if (_EVAL_ELO_FREQUENCY.value > 0
                and multi_step % _EVAL_ELO_FREQUENCY.value == 0):
            metrics.eval_elo(go_model, train_data.params, board_size)

    return train_data.params, pd.json_normalize(metrics_logs)
