"""Module for understanding the behavior of the code."""
import copy
import itertools
import re

import absl.flags
import gojax
import haiku as hk
import jax.random
import optax
import pandas as pd
from jax import numpy as jnp
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from muzero_gojax import game


def _plot_state(axis, state: jnp.ndarray):
    axis.imshow(
        state[gojax.BLACK_CHANNEL_INDEX].astype(int) - state[gojax.WHITE_CHANNEL_INDEX].astype(int),
        vmin=-1, vmax=1, cmap='Greys')


def play_against_model(go_model: hk.MultiTransformedWithState, params: optax.Params,
                       model_state: dict, absl_flags: absl.flags.FlagValues):
    """
    Deploys an interactive terminal to play against the Go model.

    :param go_model: Haiku Go model.
    :param params: Model parameters.
    :param model_state: Model state.
    :param absl_flags: Abseil flags.
    :return: None.
    """
    cap_letters = 'ABCDEFGHIJKLMNOPQRS'

    states = gojax.new_states(absl_flags.board_size)
    print(gojax.get_pretty_string(states[0]))
    rng_key = jax.random.PRNGKey(absl_flags.random_seed)
    step = 0
    while not gojax.get_ended(states):
        # Get user's move.
        re_match = re.match(r'\s*(\d+)\s+(\D+)\s*', input('Enter move (R C):'))
        while not re_match:
            re_match = re.match(r'\s*(\d+)\s+(\D+)\s*', input('Enter move (R C):'))
        row = int(re_match.group(1))
        col = cap_letters.index(re_match.group(2).upper())
        indicator_actions = gojax.action_2d_indices_to_indicator([(row, col)], states)
        states = gojax.next_states(states, indicator_actions)
        print(gojax.get_pretty_string(states[0]))
        if gojax.get_ended(states):
            break

        # Get AI's move.
        print('Model thinking...')
        rng_key = jax.random.fold_in(rng_key, step)
        states = game.sample_next_states(go_model, params, model_state, rng_key, states)
        print(gojax.get_pretty_string(states[0]))
        step += 1


def get_interesting_states(board_size: int):
    """
    Returns a set of interesting states which we would like to see how the model reacts.

    1) Empty state.
    2) Easy kill.
    """
    # 0 index is empty state.
    states = gojax.new_states(board_size, batch_size=3)

    # 1st index is easy kill at the corner.
    for i, j in [(2, 0), (0, 2), (1, 2)]:
        states = states.at[1, gojax.BLACK_CHANNEL_INDEX, i, j].set(True)
    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        states = states.at[1, gojax.WHITE_CHANNEL_INDEX, i, j].set(True)

    # Every other space is filled by black.
    for action_1d in range(0, board_size ** 2, 2):
        i = action_1d // board_size
        j = action_1d % board_size
        states = states.at[2, gojax.BLACK_CHANNEL_INDEX, i, j].set(True)

    return states


def plot_model_thoughts(go_model: hk.MultiTransformedWithState, params: optax.Params,
                        model_state: dict, states: jnp.ndarray,
                        rng_key: jax.random.KeyArray = None):
    """
    Plots a heatmap of the policy for the given state, and bar plots of the pass and value logits.

    Plots (1) the state, (2) the non-pass action logits, (3) the pass logit.
    """
    if not rng_key:
        rng_key = jax.random.PRNGKey(42)
    fig, axes = plt.subplots(nrows=len(states), ncols=4, figsize=(16, 4 * len(states)),
                             squeeze=False)
    for i, state in enumerate(states):
        state = jnp.expand_dims(state, axis=0)
        policy_logits = \
            game.get_policy_logits(go_model, params, model_state, state, rng_key).astype('float32')[
                0]
        action_logits = jnp.reshape(policy_logits[:-1], state.shape[-2:])
        axes[i, 0].set_title('State')
        _plot_state(axes[i, 0], state[0])

        axes[i, 1].set_title('Action logits')
        image = axes[i, 1].imshow(action_logits)
        fig.colorbar(image, ax=axes[i, 1])

        axes[i, 2].set_title('Action logits\nnormalized range')
        image = axes[i, 2].imshow(action_logits, vmin=-3, vmax=3)
        fig.colorbar(image, ax=axes[i, 2])

        axes[i, 3].set_title('Pass & Value logits')
        embed_model, value_model = go_model.apply[:2]
        value_logit = value_model(params, model_state, rng_key,
                                  embed_model(params, model_state, rng_key, state)[0])[0].astype(
            'float32')
        axes[i, 3].bar(['pass', 'value'], [policy_logits[-1], value_logit])


def plot_metrics(metrics_df: pd.DataFrame):
    """Plots the metrics dataframe."""
    _, axes = plt.subplots(1, 2, figsize=(8, 3))
    metrics_df.plot(ax=axes[0])
    metrics_df.plot(logy=True, ax=axes[1])
    plt.tight_layout()


def plot_trajectories(trajectories: jnp.ndarray):
    """
    Plots trajectories.

    :param trajectories: An N x T x C x B x B boolean array
    """
    nrows, ncols, _, board_size, _ = trajectories.shape
    actions1d, winner = game.get_actions_and_labels(trajectories)
    _, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    for i, j in itertools.product(range(nrows), range(ncols)):
        action_1d = actions1d[i, j - 1] if j > 0 else None
        _plot_state(axes[i, j], trajectories[i, j])
        if action_1d is not None:
            if action_1d < board_size ** 2:
                rect = patches.Rectangle(
                    xy=(action_1d % board_size - 0.5, action_1d // board_size - 0.5), width=1,
                    height=1, linewidth=2, edgecolor='g', facecolor='none')
            else:
                rect = patches.Rectangle(xy=(-0.5, -0.5), width=board_size, height=board_size,
                                         linewidth=4, edgecolor='orange', facecolor='none')
            axes[i, j].add_patch(rect)
        axes[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[i, j].yaxis.set_major_locator(MaxNLocator(integer=True))
        turn = 'W' if gojax.get_turns(jnp.expand_dims(trajectories[i, j], 0)) else 'B'
        if winner[i, j] == 1:
            won_str = 'Won'
        elif winner[i, j] == 0:
            won_str = 'Tie'
        elif winner[i, j] == -1:
            won_str = 'Lost'
        else:
            raise Exception(f'Unknown game winner value: {winner[i, j]}')
        axes[i, j].set_title(f'{turn}, {won_str}')


def _get_weights_and_biases(params: optax.Params):
    """Extracts the weights and biases of the parameters."""
    weights, biases = [], []
    for key, value in params.items():
        if isinstance(value, dict):
            subweights, subbiases = _get_weights_and_biases(value)
            weights.extend(subweights)
            biases.extend(subbiases)
        else:
            flattened_values = value.astype('float32').flatten().tolist()
            if key == 'w':
                weights.extend(flattened_values)
            elif key == 'b':
                biases.extend(flattened_values)
            else:
                print(f'WARNING: Unable to determine weight type of {key}.')

    return weights, biases


def plot_histogram_weights(params: optax.Params):
    """Plots a histogram of the weights and biases of the parameters."""
    plt.figure()
    weights, biases = _get_weights_and_biases(params)
    plt.hist(weights, label='weights', alpha=0.5, density=1)
    plt.hist(biases, label='biases', alpha=0.5, density=1)
    plt.legend()


def plot_sample_trajectores(absl_flags: absl.flags.FlagValues,
                            go_model: hk.MultiTransformedWithState, params: optax.Params,
                            model_state):
    """Plots a sample of trajectories."""
    flags_copy = copy.deepcopy(absl_flags)
    flags_copy.batch_size = 2
    flags_copy.max_num_steps = 10
    sample_traj = game.self_play(flags_copy, go_model, params, model_state, jax.random.PRNGKey(42))
    plot_trajectories(sample_traj)
