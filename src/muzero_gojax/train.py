"""Manages the MuZero training of Go models."""
import dataclasses
import functools
from datetime import datetime
from typing import Callable, Optional, Tuple

import chex
import gojax
import haiku as hk
import jax
import jax.nn
import jax.numpy as jnp
import jax.random
import optax
import pandas as pd
from absl import flags
from jax import lax

from muzero_gojax import game, losses, models, nt_utils

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
_UPDATES_PER_TRAIN_STEP = flags.DEFINE_integer(
    'updates_per_train_step', 1,
    'Number of model updates per train step to run.')
_EVAL_FREQUENCY = flags.DEFINE_integer(
    'eval_frequency', 1, 'How often to evaluate the model. '
    'Set this value to <= 0 to deactivate the JIT on the train step')

_BATCH_SIZE = flags.DEFINE_integer('batch_size', 2,
                                   'Size of the batch to train_model on.')
_TRAJECTORY_LENGTH = flags.DEFINE_integer(
    'trajectory_length', 26, 'Maximum number of game steps for Go.'
    'Usually set to 2(board_size^2).')
_SELF_PLAY_MODEL = flags.DEFINE_string(
    'self_play_model', None, 'Which model to use to generate trajectories. '
    'Defaults to using the model in training.')
_SELF_PLAY_SAMPLE_ACTION_SIZE = flags.DEFINE_integer(
    'self_play_sample_action_size', 0,
    'Number of actions to sample for policy improvement during self play.')
_UPDATE_SELF_PLAY_POLICY_FREQUENCY = flags.DEFINE_integer(
    'update_self_play_policy_frequency', 1,
    'If the self play model transform is the same, how frequently to update '
    'the self play model params. Otherwise not applicable.')


@chex.dataclass(frozen=True)
class TrainData:
    """Training data."""
    game_stats: game.GameStats = game.GameStats()
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
    schedule = optax.linear_schedule(0, _LEARNING_RATE.value,
                                     _LR_WARMUP_STEPS.value)
    return {
        'adam': optax.adam,
        'sgd': optax.sgd,
        'adamw': optax.adamw
    }[_OPTIMIZER.value](schedule)


def _sample_game_data(trajectories: game.Trajectories,
                      rng_key: jax.random.KeyArray) -> losses.GameData:
    batch_size, traj_len = trajectories.nt_states.shape[:2]
    batch_order_indices = jnp.expand_dims(jnp.arange(batch_size), axis=1)
    game_ended = nt_utils.unflatten_first_dim(
        gojax.get_ended(nt_utils.flatten_first_two_dims(
            trajectories.nt_states)), batch_size, traj_len)
    base_sample_state_logits = game_ended * float('-inf')
    base_indices = jax.random.categorical(rng_key,
                                          base_sample_state_logits,
                                          axis=1)
    select_indices = jnp.repeat(jnp.expand_dims(base_indices, axis=1),
                                repeats=2,
                                axis=1).at[:, 1].add(1)
    nk_states = trajectories.nt_states[batch_order_indices, select_indices]
    nk_actions = trajectories.nt_actions[batch_order_indices, select_indices]
    nt_player_labels = game.get_nt_player_labels(trajectories.nt_states)
    nk_player_labels = nt_player_labels[batch_order_indices, select_indices]
    return losses.GameData(nk_states=nk_states,
                           nk_actions=nk_actions,
                           nk_player_labels=nk_player_labels)


def _update_step(go_model, optimizer: optax.GradientTransformation,
                 augmented_trajectories: game.Trajectories, _: int,
                 train_data: TrainData) -> TrainData:
    rng_key, subkey = jax.random.split(train_data.rng_key)
    game_data: losses.GameData = _sample_game_data(augmented_trajectories,
                                                   subkey)
    del subkey
    rng_key, subkey = jax.random.split(rng_key)
    grads, loss_metrics = losses.compute_loss_gradients_and_metrics(
        go_model, train_data.params, game_data, subkey)
    del subkey
    params, opt_state = _update_model(grads, optimizer, train_data.params,
                                      train_data.opt_state)
    return train_data.replace(params=params,
                              opt_state=opt_state,
                              rng_key=rng_key,
                              loss_metrics=loss_metrics)


def _train_step(board_size: int,
                self_play_policy: Optional[models.PolicyModel],
                go_model: hk.MultiTransformed,
                optimizer: optax.GradientTransformation, _: int,
                train_data: TrainData) -> TrainData:
    """
    Executes a single train step comprising self-play, and an update.
    :param board_size: board size.
    :param self_play_policy: Policy to generate games.
    :param go_model: JAX-Haiku model architecture.
    :param _: ignored training step index.
    :param optimizer: Optax optimizer.
    :param train_data: Train data.
    :return:
    """
    rng_key, subkey = jax.random.split(train_data.rng_key)
    if self_play_policy is None:
        self_play_policy = models.get_policy_model(
            go_model, train_data.params, _SELF_PLAY_SAMPLE_ACTION_SIZE.value)
    trajectories = game.self_play(
        game.new_trajectories(board_size, _BATCH_SIZE.value,
                              _TRAJECTORY_LENGTH.value), self_play_policy,
        subkey)
    del subkey
    game_stats = game.get_game_stats(trajectories.nt_states)
    augmented_trajectories: game.Trajectories = game.rotationally_augment_trajectories(
        trajectories)
    _, subkey = jax.random.split(rng_key)
    return jax.lax.fori_loop(
        0, _UPDATES_PER_TRAIN_STEP.value,
        jax.tree_util.Partial(_update_step, go_model, optimizer,
                              augmented_trajectories),
        train_data.replace(game_stats=game_stats, rng_key=subkey))


def _get_initial_self_play_policy_model(
        go_model: hk.MultiTransformed,
        params: optax.Params) -> models.PolicyModel:
    if _SELF_PLAY_MODEL.value == 'random':
        print("Setting initial self play model as random.")
        policy_model = models.get_policy_model(
            models.make_random_policy_tromp_taylor_value_model(), params={})
    elif _SELF_PLAY_MODEL.value == 'tromp_taylor':
        print("Setting initial self play model as Tromp Taylor.")
        policy_model = models.get_policy_model(
            models.make_tromp_taylor_model(), params={})
    elif _SELF_PLAY_MODEL.value == 'tromp_taylor_amplified':
        print("Setting initial self play model as Tromp Taylor Amplified.")
        policy_model = models.get_policy_model(
            models.make_tromp_taylor_amplified_model(), params={})
    elif _SELF_PLAY_MODEL.value is not None and _SELF_PLAY_MODEL.value != '':
        # Load the specified model for self-play game generation.
        print(f"Loading initial self play model from {_SELF_PLAY_MODEL.value}")
        self_play_model_transform, self_play_model_params, _ = models.load_model(
            _SELF_PLAY_MODEL.value)
        policy_model = models.get_policy_model(
            self_play_model_transform, self_play_model_params,
            _SELF_PLAY_SAMPLE_ACTION_SIZE.value)
    elif _UPDATE_SELF_PLAY_POLICY_FREQUENCY.value > 1:
        # By default, use the model in training to generate self-play games.
        print(
            "Self play model will be set as current version of model in training."
        )
        policy_model = models.get_policy_model(
            go_model, params, _SELF_PLAY_SAMPLE_ACTION_SIZE.value)
    else:
        print("Self play model will be itself (None).")
        policy_model = None
    return policy_model


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
    if _TRAINING_STEPS.value <= 0:
        # Return early.
        metrics_df = pd.json_normalize([])
        return params, metrics_df

    optimizer = _get_optimizer()
    opt_state = optimizer.init(params)

    train_data = TrainData(params=params,
                           opt_state=opt_state,
                           loss_metrics=_init_loss_metrics(dtype),
                           rng_key=rng_key)
    single_train_step_fn = jax.tree_util.Partial(
        _train_step, board_size,
        _get_initial_self_play_policy_model(go_model, params), go_model,
        optimizer)

    train_history = []
    start_train_time = datetime.now().replace(microsecond=0)
    for multi_step in range(
            max(_EVAL_FREQUENCY.value, 1),
            _TRAINING_STEPS.value + max(_EVAL_FREQUENCY.value, 1),
            max(_EVAL_FREQUENCY.value, 1)):
        try:
            if _EVAL_FREQUENCY.value > 1:
                train_data = _multiple_train_steps(single_train_step_fn,
                                                   _EVAL_FREQUENCY.value,
                                                   train_data)
            else:
                train_data = single_train_step_fn(0, train_data)
        except KeyboardInterrupt:
            print("Caught keyboard interrupt. Ending training early.")
            break
        train_step_data = dataclasses.asdict(train_data.loss_metrics)
        train_step_data.update(dataclasses.asdict(train_data.game_stats))
        train_history.append(
            jax.tree_util.tree_map(lambda x: round(x.item(), 3),
                                   train_step_data))
        print(f'{(datetime.now().replace(microsecond=0) - start_train_time)} '
              f'| {multi_step}: '
              f'{train_history[-1]}')

        if (_UPDATE_SELF_PLAY_POLICY_FREQUENCY.value > 1 and
                multi_step % _UPDATE_SELF_PLAY_POLICY_FREQUENCY.value == 0):
            print(
                "Updating self play policy with deep copy of training model.")
            new_self_play_policy = models.get_policy_model(
                go_model, jax.tree_util.tree_map(jnp.copy, train_data.params),
                _SELF_PLAY_SAMPLE_ACTION_SIZE.value)
            single_train_step_fn = jax.tree_util.Partial(
                _train_step, board_size, new_self_play_policy, go_model,
                optimizer)
            print("Resetting optimizer state.")
            train_data = train_data.replace(
                opt_state=optimizer.init(train_data.params))

    metrics_df = pd.json_normalize(train_history)
    return train_data.params, metrics_df
