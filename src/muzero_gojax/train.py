"""Manages the MuZero training of Go models."""

import os
import pickle
import time
from typing import Optional
from typing import Tuple

import gojax
import haiku as hk
import jax.nn
import jax.random
import optax
import pandas as pd
from absl import flags

from muzero_gojax import game
from muzero_gojax import losses
from muzero_gojax import metrics


def update_model(grads: optax.Params, optimizer: optax.GradientTransformation, params: optax.Params,
                 opt_state: optax.OptState) -> Tuple[optax.Params, optax.OptState]:
    """Updates the model in a single train_model step."""
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


def get_optimizer(absl_flags: flags.FlagValues) -> optax.GradientTransformation:
    """Gets the JAX optimizer for the corresponding name."""
    return {'adam': optax.adam, 'sgd': optax.sgd, 'adamw': optax.adamw}[absl_flags.optimizer](
        absl_flags.learning_rate)


def train_model(go_model: hk.MultiTransformed, params: optax.Params,
                absl_flags: flags.FlagValues) -> Tuple[optax.Params, pd.DataFrame]:
    """
    Trains the model with the specified hyperparameters.

    :param go_model: JAX-Haiku model architecture.
    :param params: Model parameters.
    :param absl_flags: Abseil hyperparameter flags.
    :return: The model parameters and a metrics dataframe.
    """
    optimizer = get_optimizer(absl_flags)
    opt_state = optimizer.init(params)

    rng_key = jax.random.PRNGKey(absl_flags.rng)
    train_step_fn = jax.tree_util.Partial(train_step, absl_flags, go_model, optimizer)
    if absl_flags.use_jit:
        train_step_fn = jax.jit(train_step_fn)
    metrics_df = pd.DataFrame()
    for step in range(absl_flags.training_steps):
        rng_key, subkey = jax.random.split(rng_key)
        metrics_data, opt_state, params = train_step_fn(opt_state, params, subkey)
        del subkey
        metrics_df = pd.concat(
            (metrics_df, pd.DataFrame(jax.tree_util.tree_map(lambda x: (x.item(),), metrics_data))),
            ignore_index=True)
        if absl_flags.eval_frequency <= 0 or step % absl_flags.eval_frequency == 0:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            print(f'{timestamp} | {step}: Loss metrics: {metrics_data}')
    return params, metrics_df


def train_step(absl_flags: flags.FlagValues, go_model: hk.MultiTransformed,
               optimizer: optax.GradientTransformation, opt_state: optax.OptState,
               params: optax.Params, rng_key: jax.random.KeyArray) -> Tuple[
    metrics.Metrics, optax.OptState, optax.Params]:
    # pylint: disable=too-many-arguments
    """
    Executes a single train step comprising self-play, and an update.
    :param absl_flags: Abseil hyperparameter flags.
    :param go_model: JAX-Haiku model architecture.
    :param optimizer: Optax optimizer.
    :param opt_state: Optimizer state.
    :param params: Model parameters.
    :param rng_key: RNG key.
    :return:
    """
    if absl_flags.train_debug_print:
        jax.debug.print("Self-playing...")
    trajectories = game.self_play(absl_flags, go_model, params, rng_key)
    if absl_flags.train_debug_print:
        jax.debug.print("Computing loss gradient...")
    grads, metrics_data = losses.compute_loss_gradients_and_metrics(absl_flags, go_model, params,
                                                                    trajectories)
    if absl_flags.train_debug_print:
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
    if absl_flags.save_dir:
        model_dir = os.path.join(absl_flags.save_dir, hash_model_flags(absl_flags))
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


def init_model(go_model: hk.MultiTransformed, absl_flags: flags.FlagValues) -> optax.Params:
    """Initializes model either randomly or from laoding a previous save file."""
    rng_key = jax.random.PRNGKey(absl_flags.rng)
    if absl_flags.load_dir:
        params = load_tree_array(os.path.join(absl_flags.load_dir, 'params.npz'), dtype='bfloat16')
        print(f"Loaded parameters from '{absl_flags.load_dir}'.")
    else:
        params = go_model.init(rng_key, gojax.new_states(absl_flags.board_size, 1))
        print("Initialized parameters randomly.")
    return params
