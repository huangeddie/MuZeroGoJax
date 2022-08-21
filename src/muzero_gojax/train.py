"""Manages the MuZero training of Go models."""
import os
import pickle
from typing import Tuple

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


def update_model(absl_flags: absl.flags.FlagValues, go_model: hk.MultiTransformedWithState,
                 optimizer: optax.GradientTransformation, params: optax.Params, model_state, opt_state,
                 trajectories: jnp.ndarray):
    # pylint: disable=too-many-arguments
    """Updates the model in a single train_model step."""
    loss_fn = jax.value_and_grad(losses.compute_k_step_total_loss, argnums=2, has_aux=True)
    (total_loss, metrics_data), grads = loss_fn(absl_flags, go_model, params, model_state, trajectories,
                                                absl_flags.hypo_steps, absl_flags.temperature)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    # TODO: Optimize model state redundancy.
    return params, metrics_data['model_state'], opt_state, metrics_data


def get_optimizer(opt_name: str):
    """Gets the JAX optimizer for the corresponding name."""
    return {'adam': optax.adam, 'sgd': optax.sgd, 'adamw': optax.adamw}[opt_name]


def train_model(go_model: hk.MultiTransformedWithState, params: optax.Params, model_state: dict,
                absl_flags: absl.flags.FlagValues) -> Tuple[optax.Params, dict, pd.DataFrame]:
    """
    Trains the model with the specified hyperparameters.

    :param go_model: JAX-Haiku model architecture.
    :param params: Model parameters.
    :param model_state: Model state.
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
        # TODO: Optimize model state redundancy.
        loss_metrics, opt_state, params, model_state = train_step_fn(opt_state, params, model_state, rng_key)
        loss_metrics_copy = loss_metrics.copy()
        loss_metrics_copy.pop('model_state')
        metrics_df = pd.concat(
            (metrics_df, pd.DataFrame(jax.tree_util.tree_map(lambda x: (x.item(),), loss_metrics_copy))),
            ignore_index=True)
        print(f'{step}: Loss metrics: {loss_metrics}')
    return params, model_state, metrics_df


def train_step(absl_flags: absl.flags.FlagValues, go_model: hk.MultiTransformedWithState,
               optimizer: optax.GradientTransformation, opt_state: optax.OptState, params: optax.Params, model_state,
               rng_key: jax.random.KeyArray):
    """
    Executes a single train step comprising self-play, and an update.
    :param absl_flags: Abseil hyperparameter flags.
    :param go_model: JAX-Haiku model architecture.
    :param optimizer: Optax optimizer.
    :param opt_state: Optimizer state.
    :param params: Model parameters.
    :param model_state: Model state.
    :param rng_key: RNG key.
    :return:
    """
    trajectories = game.self_play(absl_flags, go_model, params, model_state, rng_key)
    # TODO: Optimize model state redundancy.
    params, model_state, opt_state, metrics_data = update_model(absl_flags, go_model, optimizer, params, model_state,
                                                                opt_state, trajectories)
    return metrics_data, opt_state, params, model_state


def maybe_save_model(params: optax.Params, model_state: dict, absl_flags: absl.flags.FlagValues):
    """
    Saves the parameters with a filename that is the hash of the absl_flags without the load_dir flag.

    :param params: Model parameters.
    :param model_state: Model state.
    :param absl_flags: Abseil flags.
    :return: None or the model directory.
    """
    if absl_flags.save_dir:
        model_dir = os.path.join(absl_flags.save_dir, hash_model_flags(absl_flags))
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        params_filename = os.path.join(model_dir, 'params.npz')
        with open(params_filename, 'wb') as f:
            pickle.dump(jax.tree_util.tree_map(lambda x: x.astype('float32'), params), f)
        model_state_filename = os.path.join(model_dir, 'model_state.npz')
        with open(model_state_filename, 'wb') as f:
            pickle.dump(jax.tree_util.tree_map(lambda x: x.astype('float32'), model_state), f)
        print(f"Saved model to '{model_dir}'.")
        return model_dir
    else:
        print(f"Model NOT saved.")


def hash_model_flags(absl_flags: absl.flags.FlagValues):
    """Hashes all model config related flags."""
    model_flags = ('embed_model', 'value_model', 'policy_model', 'transition_model', 'hdim')
    model_flag_values = tuple(map(lambda flag_name: str(absl_flags.get_flag_value(flag_name, '')), model_flags))
    return str(hash(':'.join(model_flag_values)))


def load_tree_array(filepath: str, dtype: str = None):
    """Loads the parameters casted into an optional type"""
    with open(filepath, 'rb') as f:
        tree = pickle.load(f)
    if dtype:
        tree = jax.tree_util.tree_map(lambda x: x.astype(dtype), tree)
    return tree


def init_model(go_model: hk.MultiTransformedWithState, absl_flags: absl.flags.FlagValues):
    """Initializes model either randomly or from laoding a previous save file."""
    rng_key = jax.random.PRNGKey(absl_flags.random_seed)
    if absl_flags.load_dir:
        params = load_tree_array(os.path.join(absl_flags.load_dir, 'params.npz'), dtype='bfloat16')
        model_state = load_tree_array(os.path.join(absl_flags.load_dir, 'model_state.npz'), dtype='bfloat16')
        print(f"Loaded parameters from '{absl_flags.load_dir}'.")
    else:
        params, model_state = go_model.init(rng_key, gojax.new_states(absl_flags.board_size, 1))
        print("Initialized parameters randomly.")
    return params, model_state
