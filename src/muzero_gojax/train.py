"""Manages the MuZero training of Go models."""
import functools
import time
from typing import Callable
from typing import NamedTuple
from typing import Tuple

import haiku as hk
import jax.nn
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
import pandas as pd
from absl import flags
from jax import lax

from muzero_gojax import game
from muzero_gojax import losses
from muzero_gojax import metrics

_RNG = flags.DEFINE_integer("rng", 42, "Random seed.")
_OPTIMIZER = flags.DEFINE_enum("optimizer", 'sgd', ['sgd', 'adam', 'adamw'], "Optimizer.")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 0.01, "Learning rate for the optimizer.")
_TRAINING_STEPS = flags.DEFINE_integer("training_steps", 10, "Number of training steps to run.")
_EVAL_FREQUENCY = flags.DEFINE_integer("eval_frequency", 1, "How often to evaluate the model.")

_BATCH_SIZE = flags.DEFINE_integer("batch_size", 2, "Size of the batch to train_model on.")
_TRAJECTORY_LENGTH = flags.DEFINE_integer("trajectory_length", 50,
                                          "Maximum number of game steps for Go."
                                          "Usually set to 2(board_size^2).")

_TRAIN_DEBUG_PRINT = flags.DEFINE_bool('train_debug_print', False,
                                       'Log stages in the train step function?')


class TrainData(NamedTuple):
    """Training data."""
    params: optax.Params = None
    opt_state: optax.OptState = None
    metrics_data: metrics.Metrics = None
    rng_key: jax.random.KeyArray = None


def _update_model(grads: optax.Params, optimizer: optax.GradientTransformation,
                  params: optax.Params, opt_state: optax.OptState) -> Tuple[
    optax.Params, optax.OptState]:
    """Updates the model in a single train_model step."""
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


def _get_optimizer() -> optax.GradientTransformation:
    """Gets the JAX optimizer for the corresponding name."""
    return {'adam': optax.adam, 'sgd': optax.sgd, 'adamw': optax.adamw}[_OPTIMIZER.value](
        _LEARNING_RATE.value)


def train_step(board_size: int, go_model: hk.MultiTransformed,
               optimizer: optax.GradientTransformation, _: int, train_data: TrainData) -> TrainData:
    """
    Executes a single train step comprising self-play, and an update.
    :param board_size: board size.
    :param go_model: JAX-Haiku model architecture.
    :param _: ignored training step index.
    :param optimizer: Optax optimizer.
    :param train_data: Train data.
    :return:
    """
    if _TRAIN_DEBUG_PRINT.value:
        jax.debug.print("Self-playing...")
    rng_key, subkey = jax.random.split(train_data.rng_key)
    trajectories = game.self_play(
        game.new_trajectories(board_size, _BATCH_SIZE.value, _TRAJECTORY_LENGTH.value), go_model,
        train_data.params, subkey)
    del subkey
    if _TRAIN_DEBUG_PRINT.value:
        jax.debug.print("Computing loss gradient...")
    augmented_trajectories: game.Trajectories = game.rotationally_augment_trajectories(trajectories)
    grads, metrics_data = losses.compute_loss_gradients_and_metrics(go_model, train_data.params,
                                                                    augmented_trajectories)
    if _TRAIN_DEBUG_PRINT.value:
        jax.debug.print("Updating model...")
    params, opt_state = _update_model(grads, optimizer, train_data.params, train_data.opt_state)
    return TrainData(params, opt_state, metrics_data, rng_key)


@functools.partial(jax.jit, static_argnums=(0,))
def _multiple_train_steps(train_step_fn: Callable, num_steps: int,
                          train_data: TrainData) -> TrainData:
    """
    Executes multiple training steps.

    This is extracted into its own JIT-ted compiled function so that the compiled function can be
    reused.
    """
    return lax.fori_loop(0, num_steps, train_step_fn, init_val=train_data)


def train_model(go_model: hk.MultiTransformed, params: optax.Params, board_size) -> Tuple[
    optax.Params, pd.DataFrame]:
    """
    Trains the model with the specified hyperparameters.

    :param go_model: JAX-Haiku model architecture.
    :param params: Model parameters.
    :param board_size: Board size.
    :return: The model parameters and a metrics dataframe.
    """
    optimizer = _get_optimizer()
    opt_state = optimizer.init(params)

    rng_key = jax.random.PRNGKey(_RNG.value)
    train_history = jnp.zeros(
        (_TRAINING_STEPS.value // _EVAL_FREQUENCY.value, len(metrics.Metrics._fields)))

    train_data = TrainData(params, opt_state, metrics.Metrics(), rng_key)
    train_step_fn = jax.tree_util.Partial(train_step, board_size, go_model, optimizer)
    for multi_step in range(1, _TRAINING_STEPS.value // _EVAL_FREQUENCY.value + 1):
        train_data = _multiple_train_steps(train_step_fn, _EVAL_FREQUENCY.value, train_data)
        train_history = train_history.at[multi_step].set(train_data.metrics_data)
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        print(f'{timestamp} | {multi_step * _EVAL_FREQUENCY.value}: {train_data.metrics_data}')

    metrics_df = pd.DataFrame(np.array(train_history), columns=list(metrics.Metrics._fields))
    return train_data.params, metrics_df


def hash_model_flags(absl_flags: flags.FlagValues) -> str:
    """Hashes all model config related flags."""
    model_flags = (
        'decode_model', 'embed_model', 'value_model', 'policy_model', 'transition_model', 'hdim',
        'embed_dim')
    model_flag_values = tuple(
        map(lambda flag_name: str(absl_flags.get_flag_value(flag_name, '')), model_flags))
    return str(hash(':'.join(model_flag_values)))
