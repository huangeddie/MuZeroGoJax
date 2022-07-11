"""Manages the MuZero training of Go models."""

import pickle

import gojax
import jax.nn
import jax.random
from jax import jit
from jax.experimental import optimizers

from muzero_gojax import game
from muzero_gojax import losses


def train_step(model_fn, opt_update, get_params, opt_state, trajectories, actions, game_winners,
               step):
    # pylint: disable=too-many-arguments
    """Updates the model in a single train_model step."""
    loss_fn = jax.value_and_grad(losses.compute_k_step_total_loss, argnums=1, has_aux=True)
    (total_loss, loss_dict), grads = loss_fn(model_fn, get_params(opt_state), trajectories, actions,
                                             game_winners)
    opt_state = opt_update(step, grads, opt_state)
    return loss_dict, opt_state


def get_optimizer(opt_name):
    """Gets the JAX optimizer for the corresponding name."""
    return {'adam': optimizers.adam, 'sgd': optimizers.sgd}[opt_name]


def train_model(model_fn, params, absl_flags):
    # pylint: disable=too-many-arguments
    """
    Trains the model with the specified hyperparameters.

    :param model_fn: JAX-Haiku model architecture.
    :param params: Model parameters.
    :param absl_flags: ABSL hyperparameter flags.
    :return: The model parameters.
    """
    self_play_fn = jax.tree_util.Partial(game.self_play, model_fn, absl_flags.batch_size,
                                         absl_flags.board_size, absl_flags.max_num_steps)
    self_play_fn = jit(self_play_fn) if absl_flags.use_jit else self_play_fn
    get_actions_and_labels_fn = jit(
        game.get_actions_and_labels) if absl_flags.use_jit else game.get_actions_and_labels

    opt_init, opt_update, get_params = get_optimizer(absl_flags.optimizer)(absl_flags.learning_rate)
    opt_state = opt_init(params)
    print(f'{sum(x.size for x in jax.tree_leaves(get_params(opt_state)))} parameters')

    train_step_fn = jax.tree_util.Partial(train_step, model_fn, opt_update, get_params)
    train_step_fn = jit(train_step_fn) if absl_flags.use_jit else train_step_fn
    rng_key = jax.random.PRNGKey(absl_flags.random_seed)
    for step in range(absl_flags.training_steps):
        print(f'{step}: Self-playing...')
        trajectories = self_play_fn(get_params(opt_state), rng_key)
        actions, game_winners = get_actions_and_labels_fn(trajectories)
        print(f'{step}: Executing training step...')
        loss_metrics, opt_state = train_step_fn(opt_state, trajectories, actions, game_winners,
                                                step)
        print(f'Loss metrics: {loss_metrics}')
    return get_params(opt_state)


def init_model(go_model, absl_flags):
    """Initializes model either randomly or from laoding a previous save file."""
    rng_key = jax.random.PRNGKey(absl_flags.random_seed)
    if absl_flags.load_path:
        params = load_params(absl_flags.load_path)
        print(f"Loaded parameters from '{absl_flags.load_path}'.")
    else:
        params = go_model.init(rng_key, gojax.new_states(absl_flags.board_size, 1))
        print(f"Initialized parameters randomly.")
    return params


def load_params(filepath, dtype=None):
    """Loads the parameters casted into an optional type"""
    with open(filepath, 'rb') as f:
        params = pickle.load(f)
    if dtype:
        params = jax.tree_map(lambda x: x.astype(dtype), params)
    return params
