"""Manages the MuZero training of Go models."""

import os
import pickle
import time
from typing import Optional
from typing import Tuple

import gojax
import haiku as hk
import jax.nn
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
import pandas as pd
from absl import flags

from muzero_gojax import game
from muzero_gojax import losses
from muzero_gojax import metrics

_RNG = flags.DEFINE_integer("rng", 42, "Random seed.")
_OPTIMIZER = flags.DEFINE_enum("optimizer", 'sgd', ['sgd', 'adam', 'adamw'], "Optimizer.")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 0.01, "Learning rate for the optimizer.")
_TRAINING_STEPS = flags.DEFINE_integer("training_steps", 10, "Number of training steps to run.")
_EVAL_FREQUENCY = flags.DEFINE_integer("eval_frequency", 0, "How often to evaluate the model.")

_BATCH_SIZE = flags.DEFINE_integer("batch_size", 2, "Size of the batch to train_model on.")
_TRAJECTORY_LENGTH = flags.DEFINE_integer("trajectory_length", 50,
                                          "Maximum number of game steps for Go."
                                          "Usually set to 2(board_size^2).")

_SAVE_DIR = flags.DEFINE_string('save_dir', None, 'File directory to save the parameters.')
_LOAD_DIR = flags.DEFINE_string('load_dir', None, 'File path to load the saved parameters.'
                                                  'Otherwise the model starts from randomly '
                                                  'initialized weights.')
_USE_JIT = flags.DEFINE_bool('use_jit', False, 'Use JIT compilation.')
_TRAIN_DEBUG_PRINT = flags.DEFINE_bool('train_debug_print', False,
                                       'Log stages in the train step function?')


def update_model(grads: optax.Params, optimizer: optax.GradientTransformation, params: optax.Params,
                 opt_state: optax.OptState) -> Tuple[optax.Params, optax.OptState]:
    """Updates the model in a single train_model step."""
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


def get_optimizer() -> optax.GradientTransformation:
    """Gets the JAX optimizer for the corresponding name."""
    return {'adam': optax.adam, 'sgd': optax.sgd, 'adamw': optax.adamw}[_OPTIMIZER.value](
        _LEARNING_RATE.value)


def train_model(go_model: hk.MultiTransformed, params: optax.Params, board_size) -> Tuple[
    optax.Params, pd.DataFrame]:
    """
    Trains the model with the specified hyperparameters.

    :param go_model: JAX-Haiku model architecture.
    :param params: Model parameters.
    :param board_size: Board size.
    :return: The model parameters and a metrics dataframe.
    """
    optimizer = get_optimizer()
    opt_state = optimizer.init(params)

    rng_key = jax.random.PRNGKey(_RNG.value)
    train_step_fn = jax.tree_util.Partial(train_step, board_size, go_model, optimizer)
    if _USE_JIT.value:
        train_step_fn = jax.jit(train_step_fn)
    train_history = jnp.zeros((_TRAINING_STEPS.value, len(metrics.Metrics._fields)))
    for step in range(_TRAINING_STEPS.value):
        rng_key, subkey = jax.random.split(rng_key)
        metrics_data, opt_state, params = train_step_fn(opt_state, params, subkey)
        del subkey
        train_history = train_history.at[step].set(metrics_data)
        if _EVAL_FREQUENCY.value <= 0 or step % _EVAL_FREQUENCY.value == 0:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            print(f'{timestamp} | {step}: {metrics_data}')

    metrics_df = pd.DataFrame(np.array(train_history), columns=list(metrics.Metrics._fields))
    return params, metrics_df


def train_step(board_size: int, go_model: hk.MultiTransformed,
               optimizer: optax.GradientTransformation, opt_state: optax.OptState,
               params: optax.Params, rng_key: jax.random.KeyArray) -> Tuple[
    metrics.Metrics, optax.OptState, optax.Params]:
    # pylint: disable=too-many-arguments
    """
    Executes a single train step comprising self-play, and an update.
    :param board_size: board size.
    :param go_model: JAX-Haiku model architecture.
    :param optimizer: Optax optimizer.
    :param opt_state: Optimizer state.
    :param params: Model parameters.
    :param rng_key: RNG key.
    :return:
    """
    if _TRAIN_DEBUG_PRINT.value:
        jax.debug.print("Self-playing...")

    trajectories = game.self_play(
        game.new_trajectories(board_size, _BATCH_SIZE.value, _TRAJECTORY_LENGTH.value), go_model,
        params, rng_key)
    if _TRAIN_DEBUG_PRINT.value:
        jax.debug.print("Computing loss gradient...")
    grads, metrics_data = losses.compute_loss_gradients_and_metrics(go_model, params, trajectories)
    if _TRAIN_DEBUG_PRINT.value:
        jax.debug.print("Updating model...")
    params, opt_state = update_model(grads, optimizer, params, opt_state)
    return metrics_data, opt_state, params


def maybe_save_model(params: optax.Params, absl_flags: flags.FlagValues) -> Optional[str]:
    """
    Saves the parameters with a filename that is the hash of the flags.

    :param params: Model parameters.
    :param absl_flags: Abseil flags.
    :return: None or the model directory.
    """
    if _SAVE_DIR.value:
        model_dir = os.path.join(_SAVE_DIR.value, hash_model_flags(absl_flags))
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        params_filename = os.path.join(model_dir, 'params.npz')
        with open(params_filename, 'wb') as params_file:
            pickle.dump(jax.tree_util.tree_map(lambda x: x.astype('float32'), params), params_file)
        print(f"Saved model to '{model_dir}'.")
        return model_dir
    print("Model NOT saved.")
    return None


def hash_model_flags(absl_flags: flags.FlagValues) -> str:
    """Hashes all model config related flags."""
    model_flags = (
        'embed_model', 'value_model', 'policy_model', 'transition_model', 'hdim', 'embed_dim')
    model_flag_values = tuple(
        map(lambda flag_name: str(absl_flags.get_flag_value(flag_name, '')), model_flags))
    return str(hash(':'.join(model_flag_values)))


def load_tree_array(filepath: str, dtype: str = None) -> dict:
    """Loads the parameters casted into an optional type"""
    with open(filepath, 'rb') as file_array:
        tree = pickle.load(file_array)
    if dtype:
        tree = jax.tree_util.tree_map(lambda x: x.astype(dtype), tree)
    return tree


def init_model(go_model: hk.MultiTransformed, board_size: int) -> optax.Params:
    """Initializes model either randomly or from laoding a previous save file."""
    rng_key = jax.random.PRNGKey(_RNG.value)
    if _LOAD_DIR.value:
        params = load_tree_array(os.path.join(_LOAD_DIR.value, 'params.npz'), dtype='bfloat16')
        print(f"Loaded parameters from '{_LOAD_DIR.value}'.")
    else:
        params = go_model.init(rng_key, gojax.new_states(board_size, 1))
        print("Initialized parameters randomly.")
    return params
