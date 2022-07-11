"""Manages the MuZero training of Go models."""

import os
import pickle

import absl.flags
import gojax
import haiku as hk
import jax.nn
import jax.numpy as jnp
import jax.random
import optax
from jax import jit

from muzero_gojax import game
from muzero_gojax import losses


def train_step(go_model: hk.MultiTransformed, optimizer: optax.GradientTransformation,
               params: optax.Params, opt_state, trajectories: jnp.ndarray, actions: jnp.ndarray,
               game_winners: jnp.ndarray, step: int):
    # pylint: disable=too-many-arguments
    """Updates the model in a single train_model step."""
    loss_fn = jax.value_and_grad(losses.compute_k_step_total_loss, argnums=1, has_aux=True)
    (total_loss, loss_dict), grads = loss_fn(go_model, params, trajectories, actions, game_winners)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_dict


def get_optimizer(opt_name: str):
    """Gets the JAX optimizer for the corresponding name."""
    return {'adam': optax.adam, 'sgd': optax.sgd}[opt_name]


def train_model(go_model: hk.MultiTransformed, params: optax.Params,
                absl_flags: absl.flags.FlagValues):
    # pylint: disable=too-many-arguments
    """
    Trains the model with the specified hyperparameters.

    :param go_model: JAX-Haiku model architecture.
    :param params: Model parameters.
    :param absl_flags: ABSL hyperparameter flags.
    :return: The model parameters.
    """
    self_play_fn = jax.tree_util.Partial(game.self_play, go_model, absl_flags.batch_size,
                                         absl_flags.board_size, absl_flags.max_num_steps)
    self_play_fn = jit(self_play_fn) if absl_flags.use_jit else self_play_fn
    get_actions_and_labels_fn = jit(
        game.get_actions_and_labels) if absl_flags.use_jit else game.get_actions_and_labels

    optimizer = get_optimizer(absl_flags.optimizer)(absl_flags.learning_rate)
    opt_state = optimizer.init(params)
    print(f'{sum(x.size for x in jax.tree_leaves(params))} parameters')

    train_step_fn = jax.tree_util.Partial(train_step, go_model, optimizer)
    train_step_fn = jit(train_step_fn) if absl_flags.use_jit else train_step_fn
    rng_key = jax.random.PRNGKey(absl_flags.random_seed)
    for step in range(absl_flags.training_steps):
        print(f'{step}: Self-playing...')
        trajectories = self_play_fn(params, rng_key)
        actions, game_winners = get_actions_and_labels_fn(trajectories)
        print(f'{step}: Executing training step...')
        params, opt_state, loss_metrics = train_step_fn(params, opt_state, trajectories, actions,
                                                        game_winners, step)
        print(f'Loss metrics: {loss_metrics}')
    return params


def maybe_save_model(params: optax.Params, absl_flags: absl.flags.FlagValues):
    """
    Saves the parameters with a filename that is the hash of the absl_flags

    :param params: Dictionary of parameters.
    :param absl_flags: ABSL flags.
    :return: None.
    """
    if absl_flags.save_dir:
        filename = os.path.join(absl_flags.save_dir,
                                str(hash(absl_flags.flags_into_string())) + '.npz')
        with open(filename, 'wb') as f:
            pickle.dump(jax.tree_map(lambda x: x.astype('float32'), params), f)
        print(f"Saved model to '{filename}'.")
        return filename
    else:
        print(f"Model NOT saved.")


def load_params(filepath: str, dtype: str = None):
    """Loads the parameters casted into an optional type"""
    with open(filepath, 'rb') as f:
        params = pickle.load(f)
    if dtype:
        params = jax.tree_map(lambda x: x.astype(dtype), params)
    return params


def init_model(go_model: hk.MultiTransformed, absl_flags: absl.flags.FlagValues):
    """Initializes model either randomly or from laoding a previous save file."""
    rng_key = jax.random.PRNGKey(absl_flags.random_seed)
    if absl_flags.load_path:
        params = load_params(absl_flags.load_path)
        print(f"Loaded parameters from '{absl_flags.load_path}'.")
    else:
        params = go_model.init(rng_key, gojax.new_states(absl_flags.board_size, 1))
        print(f"Initialized parameters randomly.")
    return params
