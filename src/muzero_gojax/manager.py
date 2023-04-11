"""Manages the MuZero training of Go models."""
import dataclasses
from typing import Optional, Tuple

import chex
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

_SELF_PLAY_MODEL = flags.DEFINE_string(
    'self_play_model', None, 'Which model to use to generate trajectories. '
    'Defaults to using the model in training.')

_EVAL_ELO_FREQUENCY = flags.DEFINE_integer(
    'eval_elo_frequency', 0,
    'Every N training steps, evaluate the model against the benchmarks.')
_SAVE_MODEL_FREQUENCY = flags.DEFINE_integer(
    'save_model_frequency', 0, 'Every N training steps, save the model.')


def _get_optimizer() -> optax.GradientTransformation:
    """Gets the JAX optimizer for the corresponding name."""
    schedule = optax.linear_schedule(0, _LEARNING_RATE.value,
                                     _LR_WARMUP_STEPS.value)
    return {
        'adam': optax.adam,
        'sgd': optax.sgd,
        'adamw': optax.adamw
    }[_OPTIMIZER.value](schedule)


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
            train.SELF_PLAY_SAMPLE_ACTION_SIZE.value)
    else:
        logger.log("Self play model will be itself (None).")
        policy_model = None
    return policy_model


def _init_loss_metrics(dtype: str) -> losses.LossMetrics:
    """Initializes the train metrics with zeros with the dtype."""
    return losses.LossMetrics(
        area_loss=jnp.zeros((), dtype=dtype),
        area_acc=jnp.zeros((), dtype=dtype),
        value_loss=jnp.zeros((), dtype=dtype),
        value_acc=jnp.zeros((), dtype=dtype),
        policy_loss=jnp.zeros((), dtype=dtype),
        policy_acc=jnp.zeros((), dtype=dtype),
        policy_entropy=jnp.zeros((), dtype=dtype),
        hypo_area_loss=jnp.zeros((), dtype=dtype),
        hypo_area_acc=jnp.zeros((), dtype=dtype),
        hypo_value_loss=jnp.zeros((), dtype=dtype),
        hypo_value_acc=jnp.zeros((), dtype=dtype),
    )


def _init_game_stats(dtype: str) -> game.GameStats:
    """Initializes the game stats with zeros with the dtype."""
    return game.GameStats(avg_game_length=jnp.zeros((), dtype=dtype),
                          black_win_pct=jnp.zeros((), dtype=dtype),
                          tie_pct=jnp.zeros((), dtype=dtype),
                          white_win_pct=jnp.zeros((), dtype=dtype),
                          piece_collision_rate=jnp.zeros((), dtype=dtype),
                          pass_rate=jnp.zeros((), dtype=dtype))


def _get_train_step_dict(step: int, train_data: train.TrainData) -> dict:
    if train.PMAP.value:
        train_data = jax.tree_map(lambda x: x[0], train_data)
    train_step_dict = dataclasses.asdict(train_data.loss_metrics)
    train_step_dict.update(dataclasses.asdict(train_data.game_stats))
    train_step_dict = jax.tree_map(lambda x: round(x.item(), 3),
                                   train_step_dict)
    train_step_dict['step'] = step
    return train_step_dict


def _log_train_step_dict(train_step_dict: dict):
    if not _LOG_LOSS_VALUES.value:
        train_step_dict = {
            k: v
            for k, v in train_step_dict.items() if not k.endswith('loss')
        }
    logger.log(f'{train_step_dict["step"]}: {train_step_dict}')


def _train_step_post_process(go_model, all_models_build_config, save_dir,
                             train_data, multi_step):
    train_step_dict = _get_train_step_dict(multi_step, train_data)
    _log_train_step_dict(train_step_dict)

    if (_SAVE_MODEL_FREQUENCY.value > 0
            and multi_step % _SAVE_MODEL_FREQUENCY.value == 0
            and save_dir is not None):
        logger.log(f'Saving model to {save_dir}')
        models.save_model(train_data.params, all_models_build_config, save_dir)

    if (_EVAL_ELO_FREQUENCY.value > 0
            and multi_step % _EVAL_ELO_FREQUENCY.value == 0):
        eval_params = (jax.tree_map(lambda x: x[0], train_data.params)
                       if train.PMAP.value else train_data.params)
        eval_dict = metrics.eval_elo(
            go_model, eval_params,
            all_models_build_config.model_build_config.board_size)
        train_step_dict.update(eval_dict)
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

    train_data = train.TrainData(
        params=params,
        opt_state=opt_state,
        loss_metrics=_init_loss_metrics(
            all_models_build_config.model_build_config.dtype),
        rng_key=rng_key,
        game_stats=_init_game_stats(
            all_models_build_config.model_build_config.dtype))
    if train.PMAP.value:
        train_data = jax.device_put_replicated(train_data, jax.local_devices())
        train_data = train_data.replace(
            rng_key=jax.random.split(rng_key, jax.device_count()))
    self_play_policy = _get_self_play_policy_model()
    metrics_logs = []
    multi_train_step_fn = train.get_multi_step_fn(
        board_size, self_play_policy, go_model, optimizer,
        _LOG_TRAINING_FREQUENCY.value)
    for multi_step in range(
            max(_LOG_TRAINING_FREQUENCY.value, 1),
            _TRAINING_STEPS.value + max(_LOG_TRAINING_FREQUENCY.value, 1),
            max(_LOG_TRAINING_FREQUENCY.value, 1)):
        try:
            train_data = multi_train_step_fn(train_data)
        except KeyboardInterrupt:
            logger.log("Caught keyboard interrupt. Ending training early.")
            break
        train_step_dict = _train_step_post_process(go_model,
                                                   all_models_build_config,
                                                   save_dir, train_data,
                                                   multi_step)

        metrics_logs.append(train_step_dict)

    params = train_data.params
    if train.PMAP.value:
        # Check the params are the same on all devices.
        first_params = jax.tree_map(lambda x: x[0], params)
        for i in range(1, jax.device_count()):
            other_device_params = jax.tree_map(lambda x: x[i], params)
            chex.assert_trees_all_equal(first_params, other_device_params)
        # Update params to be the first device's params.
        params = first_params

    return params, pd.json_normalize(metrics_logs).set_index('step')
