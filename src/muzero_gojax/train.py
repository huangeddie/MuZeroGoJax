"""Manages the MuZero training of Go models."""
import copy
import os
import pickle

import absl.flags
import gojax
import haiku as hk
import jax.nn
import jax.numpy as jnp
import jax.random
import optax
import pandas as pd

from muzero_gojax import game
from muzero_gojax import losses


def update_model(absl_flags: absl.flags.FlagValues, go_model: hk.MultiTransformed,
                 optimizer: optax.GradientTransformation, params: optax.Params, opt_state, trajectories: jnp.ndarray):
    # pylint: disable=too-many-arguments
    """Updates the model in a single train_model step."""
    loss_fn = jax.value_and_grad(losses.compute_k_step_total_loss, argnums=1, has_aux=True)
    (total_loss, loss_dict), grads = loss_fn(go_model, params, trajectories, absl_flags.hypo_steps,
                                             absl_flags.temperature)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_dict


def get_optimizer(opt_name: str):
    """Gets the JAX optimizer for the corresponding name."""
    return {'adam': optax.adam, 'sgd': optax.sgd, 'adamw': optax.adamw}[opt_name]


def train_model(go_model: hk.MultiTransformed, params: optax.Params, absl_flags: absl.flags.FlagValues):
    """
    Trains the model with the specified hyperparameters.

    :param go_model: JAX-Haiku model architecture.
    :param params: Model parameters.
    :param absl_flags: Abseil hyperparameter flags.
    :return: The model parameters and a metrics dataframe.
    """
    optimizer = get_optimizer(absl_flags.optimizer)(absl_flags.learning_rate)
    opt_state = optimizer.init(params)
    print(f'{sum(x.size for x in jax.tree_util.tree_leaves(params))} parameters')

    rng_key = jax.random.PRNGKey(absl_flags.random_seed)
    train_step_fn = jax.tree_util.Partial(train_step, absl_flags, go_model, optimizer)
    if absl_flags.use_jit:
        train_step_fn = jax.jit(train_step_fn)
    metrics_df = pd.DataFrame()
    for step in range(absl_flags.training_steps):
        rng_key = jax.random.fold_in(rng_key, step)
        loss_metrics, opt_state, params = train_step_fn(opt_state, params, rng_key)
        metrics_df = pd.concat((metrics_df, pd.DataFrame(jax.tree_util.tree_map(lambda x: (x.item(),), loss_metrics))),
                               ignore_index=True)
        print(f'{step}: Loss metrics: {loss_metrics}')
    return params, metrics_df


def train_step(absl_flags: absl.flags.FlagValues, go_model: hk.MultiTransformed,
               optimizer: optax.GradientTransformation, opt_state: optax.OptState, params: optax.Params,
               rng_key: jax.random.KeyArray):
    """
    Executes a single train step comprising of self-play, and an update. 
    :param absl_flags: Abseil hyperparameter flags.
    :param go_model: JAX-Haiku model architecture.
    :param optimizer: Optax optimizer.
    :param opt_state: Optimizer state.
    :param params: Model parameters.
    :param rng_key: RNG key.
    :return:
    """
    trajectories = game.self_play(absl_flags, go_model, params, rng_key)
    params, opt_state, loss_metrics = update_model(absl_flags, go_model, optimizer, params, opt_state, trajectories)
    return loss_metrics, opt_state, params


def maybe_save_model(params: optax.Params, absl_flags: absl.flags.FlagValues):
    """
    Saves the parameters with a filename that is the hash of the absl_flags without the load_path flag.

    :param params: Dictionary of parameters.
    :param absl_flags: Abseil flags.
    :return: None.
    """
    if absl_flags.save_dir:
        filename = os.path.join(absl_flags.save_dir, hash_flags(absl_flags))
        with open(filename, 'wb') as f:
            pickle.dump(jax.tree_util.tree_map(lambda x: x.astype('float32'), params), f)
        print(f"Saved model to '{filename}'.")
        return filename
    else:
        print(f"Model NOT saved.")


def hash_flags(absl_flags: absl.flags.FlagValues):
    """
    Hashes the flags without the load_path flag.

    Does not modify the given flags.
    """
    absl_flags = copy.deepcopy(absl_flags)
    absl_flags.remove_flag_values(['load_path'])
    return str(hash(absl_flags.flags_into_string())) + '.npz'


def load_params(filepath: str, dtype: str = None):
    """Loads the parameters casted into an optional type"""
    with open(filepath, 'rb') as f:
        params = pickle.load(f)
    if dtype:
        params = jax.tree_util.tree_map(lambda x: x.astype(dtype), params)
    return params


def init_model(go_model: hk.MultiTransformed, absl_flags: absl.flags.FlagValues):
    """Initializes model either randomly or from laoding a previous save file."""
    rng_key = jax.random.PRNGKey(absl_flags.random_seed)
    if absl_flags.load_path:
        params = load_params(absl_flags.load_path, dtype='bfloat16')
        print(f"Loaded parameters from '{absl_flags.load_path}'.")
    else:
        params = go_model.init(rng_key, gojax.new_states(absl_flags.board_size, 1))
        print(f"Initialized parameters randomly.")
    return params
