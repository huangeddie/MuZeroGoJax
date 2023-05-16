"""Manages the MuZero training of Go models."""
import dataclasses
import itertools
from typing import Optional, Tuple

import haiku as hk
import jax
import jax.nn
import jax.numpy as jnp
import jax.random
import optax
import pandas as pd
from absl import flags

from muzero_gojax import game, logger, losses, metrics, models, train

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
_LOG_TRAINING_FREQUENCY = flags.DEFINE_integer(
    'log_training_frequency', 1, 'How often to log the training steps. '
    'Steps within the frequency are JIT-ed. '
    'Set this value to <= 0 to deactivate the JIT on the train step')
_LOG_LOSS_VALUES = flags.DEFINE_bool('log_loss_values', False,
                                     'Whether to log loss values.')
_EVAL_ELO_FREQUENCY = flags.DEFINE_integer(
    'eval_elo_frequency', 0,
    'Every N training steps, evaluate the model against the benchmarks.')
_SAVE_MODEL_FREQUENCY = flags.DEFINE_integer(
    'save_model_frequency', 0, 'Every N training steps, save the model.')
PMAP = flags.DEFINE_bool('pmap', False, 'Whether to use pmap for training.')


def _get_optimizer() -> optax.GradientTransformation:
    """Gets the JAX optimizer for the corresponding name."""
    schedule = optax.linear_schedule(0, _LEARNING_RATE.value,
                                     _LR_WARMUP_STEPS.value)
    return {
        'adam': optax.adam,
        'sgd': optax.sgd,
        'adamw': optax.adamw
    }[_OPTIMIZER.value](schedule)


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


def _get_train_step_dict(step: int,
                         single_shard_train_data: train.TrainData) -> dict:
    train_step_dict = dataclasses.asdict(single_shard_train_data.loss_metrics)
    train_step_dict.update(
        dataclasses.asdict(single_shard_train_data.game_stats))
    train_step_dict = jax.tree_map(lambda x: round(x.item(), 3),
                                   train_step_dict)
    train_step_dict['step'] = step
    return train_step_dict


def _train_step_post_process(go_model, all_models_build_config, save_dir,
                             single_shard_train_data, multi_step):
    train_step_dict = _get_train_step_dict(multi_step, single_shard_train_data)
    if not _LOG_LOSS_VALUES.value:
        train_step_dict_to_log = {
            k: v
            for k, v in train_step_dict.items() if not k.endswith('loss')
        }
    else:
        train_step_dict_to_log = train_step_dict
    logger.log(f'{train_step_dict_to_log["step"]}: {train_step_dict_to_log}')

    if multi_step <= 0:
        return train_step_dict
    if (_SAVE_MODEL_FREQUENCY.value > 0
            and multi_step % _SAVE_MODEL_FREQUENCY.value == 0
            and save_dir is not None):
        logger.log(f'Saving model to {save_dir}')
        models.save_model(single_shard_train_data.params,
                          all_models_build_config, save_dir)

    if (_EVAL_ELO_FREQUENCY.value > 0
            and multi_step % _EVAL_ELO_FREQUENCY.value == 0):
        train_step_dict.update(
            metrics.eval_elo(
                go_model, single_shard_train_data.params,
                all_models_build_config.model_build_config.board_size))
    return train_step_dict


def train_model(
        go_model: hk.MultiTransformed,
        params: optax.Params,
        all_models_build_config: models.AllModelsBuildConfig,
        rng_key: jax.random.KeyArray,
        save_dir: Optional[str] = None) -> Tuple[optax.Params, pd.DataFrame]:
    """Trains the model with the specified hyperparameters.

    Internally, this function will 
    1. Create a new optimizer and initialize the training data. 
    2. Train the model for a number of steps. 
        2.1. For each step, it will sample a batch of games, 
        train the model on the batch, and update the optimizer state. 
        2.2. Periodically, it will log the training metrics and save the model.
    
    Args:
        go_model: The model to train.
        params: The initial parameters of the model.
        all_models_build_config: The build config for the entire model.
        rng_key: The random key to use for the training.
        save_dir: The directory to save the model to.

    Returns:
        The trained parameters and a dataframe with the training metrics.
    """
    if _TRAINING_STEPS.value <= 0:
        # Return early.
        return params, pd.json_normalize([])

    optimizer = _get_optimizer()
    opt_state = optimizer.init(params)
    board_size = all_models_build_config.model_build_config.board_size

    single_shard_train_data = train.TrainData(
        params=params,
        opt_state=opt_state,
        loss_metrics=_init_loss_metrics(),
        rng_key=rng_key,
        game_stats=_init_game_stats())
    if PMAP.value:
        train_data = jax.device_put_replicated(single_shard_train_data,
                                               jax.local_devices())
        train_data = train_data.replace(
            rng_key=jax.random.split(rng_key, jax.device_count()))
    else:
        train_data = single_shard_train_data
    metrics_logs = []

    single_train_step_fn = train.get_multi_step_fn(board_size,
                                                   go_model,
                                                   optimizer,
                                                   num_steps=1,
                                                   pmap=PMAP.value)
    multi_train_step_fn = train.get_multi_step_fn(
        board_size, go_model, optimizer, _LOG_TRAINING_FREQUENCY.value,
        PMAP.value)
    for multi_step in itertools.chain(
            range(1),
            range(
                max(_LOG_TRAINING_FREQUENCY.value, 1),
                _TRAINING_STEPS.value + max(_LOG_TRAINING_FREQUENCY.value, 1),
                max(_LOG_TRAINING_FREQUENCY.value, 1))):
        try:
            if multi_step <= 0 or _TRAINING_STEPS.value <= 1:
                train_data = single_train_step_fn(train_data)
            else:
                train_data = multi_train_step_fn(train_data)
            if PMAP.value:
                single_shard_train_data = jax.tree_map(lambda x: x[0],
                                                       train_data)
            else:
                single_shard_train_data = train_data
        except KeyboardInterrupt:
            logger.log("Caught keyboard interrupt. Ending training early.")
            break
        try:
            train_step_dict = _train_step_post_process(
                go_model, all_models_build_config, save_dir,
                single_shard_train_data, multi_step)

            metrics_logs.append(train_step_dict)
        except Exception as exception:
            logger.log(f'Error in train step post process: {exception}')
            break

    if save_dir is not None:
        try:
            models.save_model(single_shard_train_data.params,
                              all_models_build_config, save_dir)
        except Exception as exception:
            logger.log(f'Error saving model: {exception}')
    return single_shard_train_data.params, pd.json_normalize(
        metrics_logs).set_index('step')
