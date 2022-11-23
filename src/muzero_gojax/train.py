"""Manages the MuZero training of Go models."""
import dataclasses
import functools
import time
from typing import Callable, Tuple

import chex
import haiku as hk
import jax.nn
import jax.numpy as jnp
import jax.random
import optax
import pandas as pd
from absl import flags
from jax import lax

from muzero_gojax import game, losses, models

_OPTIMIZER = flags.DEFINE_enum('optimizer', 'sgd', ['sgd', 'adam', 'adamw'],
                               'Optimizer.')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 0.01,
                                    'Learning rate for the optimizer.')
_TRAINING_STEPS = flags.DEFINE_integer('training_steps', 10,
                                       'Number of training steps to run.')
_EVAL_FREQUENCY = flags.DEFINE_integer(
    'eval_frequency', 1, 'How often to evaluate the model. '
    'Set this value to <= 0 to deactivate the JIT on the train step')

_BATCH_SIZE = flags.DEFINE_integer('batch_size', 2,
                                   'Size of the batch to train_model on.')
_TRAJECTORY_LENGTH = flags.DEFINE_integer(
    'trajectory_length', 26, 'Maximum number of game steps for Go.'
    'Usually set to 2(board_size^2).')
_SELF_PLAY_MODEL = flags.DEFINE_enum(
    'self_play_model', 'self', ['random', 'self'],
    'Which model to use to generate trajectories.')


@chex.dataclass(frozen=True)
class TrainData:
    """Training data."""
    params: optax.Params = None
    opt_state: optax.OptState = None
    loss_metrics: losses.LossMetrics = None
    rng_key: jax.random.KeyArray = None


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
    return {
        'adam': optax.adam,
        'sgd': optax.sgd,
        'adamw': optax.adamw
    }[_OPTIMIZER.value](_LEARNING_RATE.value)


def _train_step(board_size: int, go_model: hk.MultiTransformed,
                optimizer: optax.GradientTransformation, _: int,
                train_data: TrainData) -> TrainData:
    """
    Executes a single train step comprising self-play, and an update.
    :param board_size: board size.
    :param go_model: JAX-Haiku model architecture.
    :param _: ignored training step index.
    :param optimizer: Optax optimizer.
    :param train_data: Train data.
    :return:
    """
    rng_key, subkey = jax.random.split(train_data.rng_key)
    self_play_model = {
        'random': models.make_random_model(),
        'self': go_model
    }[_SELF_PLAY_MODEL.value]
    trajectories = game.self_play(
        game.new_trajectories(board_size, _BATCH_SIZE.value,
                              _TRAJECTORY_LENGTH.value), self_play_model,
        train_data.params, subkey)
    del subkey
    augmented_trajectories: game.Trajectories = game.rotationally_augment_trajectories(
        trajectories)
    rng_key, subkey = jax.random.split(train_data.rng_key)
    grads, loss_metrics = losses.compute_loss_gradients_and_metrics(
        go_model, train_data.params, augmented_trajectories, subkey)
    del subkey
    params, opt_state = _update_model(grads, optimizer, train_data.params,
                                      train_data.opt_state)
    return TrainData(
        params=params,
        opt_state=opt_state,
        loss_metrics=loss_metrics,
        rng_key=rng_key,
    )


@functools.partial(jax.jit, static_argnums=(0, ))
def _multiple_train_steps(train_step_fn: Callable, num_steps: int,
                          train_data: TrainData) -> TrainData:
    """
    Executes multiple training steps.

    This is extracted into its own JIT-ted compiled function so that the compiled function can be
    reused.
    """
    return lax.fori_loop(0, num_steps, train_step_fn, init_val=train_data)


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
        black_wins=-jnp.ones((), dtype=dtype),
        ties=-jnp.ones((), dtype=dtype),
        white_wins=-jnp.ones((), dtype=dtype),
        avg_game_length=jnp.zeros((), dtype=dtype),
    )


def train_model(
        go_model: hk.MultiTransformed, params: optax.Params, board_size: int,
        dtype: str,
        rng_key: jax.random.KeyArray) -> Tuple[optax.Params, pd.DataFrame]:
    """
    Trains the model with the specified hyperparameters.

    :param go_model: JAX-Haiku model architecture.
    :param params: Model parameters.
    :param board_size: Board size.
    :return: The model parameters and a metrics dataframe.
    """
    optimizer = _get_optimizer()
    opt_state = optimizer.init(params)

    train_history = []

    train_data = TrainData(params=params,
                           opt_state=opt_state,
                           loss_metrics=_init_loss_metrics(dtype),
                           rng_key=rng_key)
    train_step_fn = jax.tree_util.Partial(_train_step, board_size, go_model,
                                          optimizer)
    for multi_step in range(
            max(_TRAINING_STEPS.value // _EVAL_FREQUENCY.value, 1)):
        if _EVAL_FREQUENCY.value > 0:
            train_data = _multiple_train_steps(
                train_step_fn, min(_EVAL_FREQUENCY.value,
                                   _TRAINING_STEPS.value), train_data)
        else:
            train_data = train_step_fn(0, train_data)
        train_history.append(
            jax.tree_util.tree_map(lambda x: round(x.item(), 3),
                                   dataclasses.asdict(
                                       train_data.loss_metrics)))
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        print(f'{timestamp} | {(multi_step + 1) * _EVAL_FREQUENCY.value}: '
              f'{train_history[-1]}')

    metrics_df = pd.json_normalize(train_history)
    return train_data.params, metrics_df


def hash_model_flags(absl_flags: flags.FlagValues) -> str:
    """Hashes all model config related flags."""
    model_flags = ('decode_model', 'embed_model', 'value_model',
                   'policy_model', 'transition_model', 'hdim', 'embed_dim')
    model_flag_values = tuple(
        map(lambda flag_name: str(absl_flags.get_flag_value(flag_name, '')),
            model_flags))
    return str(hash(':'.join(model_flag_values)))
