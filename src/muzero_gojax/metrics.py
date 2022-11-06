"""Module for understanding the behavior of the code."""
import itertools
import re

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
from muzero_gojax import models
from muzero_gojax import nt_utils
from muzero_gojax import data


def _plot_state(axis, state: jnp.ndarray):
    axis.imshow(state[gojax.BLACK_CHANNEL_INDEX].astype(int) -
                state[gojax.WHITE_CHANNEL_INDEX].astype(int),
                vmin=-1,
                vmax=1,
                cmap='Greys')
    board_size = state.shape[-1]
    edgecolor = 'yellow' if jnp.alltrue(
        state[gojax.TURN_CHANNEL_INDEX]) else 'blue'
    turn_rect = patches.Rectangle(xy=(-0.5, -0.5),
                                  width=board_size,
                                  height=board_size,
                                  linewidth=12,
                                  edgecolor=edgecolor,
                                  facecolor='none')
    axis.add_patch(turn_rect)
    if jnp.alltrue(state[gojax.END_CHANNEL_INDEX]):
        end_rect = patches.Rectangle(xy=(-0.5, -0.5),
                                     width=board_size,
                                     height=board_size,
                                     linewidth=6,
                                     edgecolor='maroon',
                                     facecolor='none')
        axis.add_patch(end_rect)
    elif jnp.alltrue(state[gojax.PASS_CHANNEL_INDEX]):
        pass_rect = patches.Rectangle(xy=(-0.5, -0.5),
                                      width=board_size,
                                      height=board_size,
                                      linewidth=6,
                                      edgecolor='orange',
                                      facecolor='none')
        axis.add_patch(pass_rect)


def play_against_model(go_model: hk.MultiTransformed, params: optax.Params,
                       board_size):
    """
    Deploys an interactive terminal to play against the Go model.

    :param go_model: Haiku Go model.
    :param params: Model parameters.
    :param board_size: Board size.
    :return: None.
    """
    cap_letters = 'ABCDEFGHIJKLMNOPQRS'

    states = gojax.new_states(board_size)
    gojax.print_state(states[0])
    rng_key = jax.random.PRNGKey(seed=42)
    step = 0
    while not gojax.get_ended(states):
        # Get user's move.
        re_match = re.match(r'\s*(\d+)\s+(\D+)\s*', input('Enter move (R C):'))
        while not re_match:
            re_match = re.match(r'\s*(\d+)\s+(\D+)\s*',
                                input('Enter move (R C):'))
        row = int(re_match.group(1))
        col = cap_letters.index(re_match.group(2).upper())
        indicator_actions = gojax.action_2d_to_indicator([(row, col)], states)
        states = gojax.next_states(states, indicator_actions)
        gojax.print_state(states[0])
        if gojax.get_ended(states):
            break

        # Get AI's move.
        print('Model thinking...')
        rng_key = jax.random.fold_in(rng_key, step)
        states = game.sample_actions_and_next_states(go_model, params, rng_key,
                                                     states)
        gojax.print_state(states[0])
        step += 1


def get_interesting_states(board_size: int):
    """Returns a set of interesting states which we would like to see how the model reacts."""
    # pylint: disable=too-many-branches
    # Empty state.
    batch_index = 0
    states = gojax.new_states(board_size, batch_size=100)

    # Black piece in the middle, and it's white's turn.
    batch_index += 1
    states = states.at[batch_index, gojax.BLACK_CHANNEL_INDEX, board_size // 2,
                       board_size // 2].set(True)
    states = states.at[batch_index,
                       gojax.TURN_CHANNEL_INDEX].set(gojax.WHITES_TURN)

    # Easy kill at the corner.
    batch_index += 1
    for i, j in [(1, 0), (0, 1), (1, 2)]:
        states = states.at[batch_index, gojax.WHITE_CHANNEL_INDEX, i,
                           j].set(True)
    states = states.at[batch_index, gojax.BLACK_CHANNEL_INDEX, 1, 1].set(True)
    states = states.at[batch_index,
                       gojax.TURN_CHANNEL_INDEX].set(gojax.WHITES_TURN)

    # Every other space is filled by black.
    batch_index += 1
    for action_1d in range(0, board_size**2):
        i = action_1d // board_size
        j = action_1d % board_size
        if (i + j) % 2 == 0:
            states = states.at[batch_index, gojax.BLACK_CHANNEL_INDEX, i,
                               j].set(True)

    # Every other space is filled by white, while it's black's turn.
    batch_index += 1
    for action_1d in range(1, board_size**2):
        i = action_1d // board_size
        j = action_1d % board_size
        if (i + j + 1) % 2 == 0:
            states = states.at[batch_index, gojax.WHITE_CHANNEL_INDEX, i,
                               j].set(True)

    # Black is surrounded but has two holes. It should not fill either hole.
    if board_size >= 4:
        batch_index += 1
        for i, j in [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]:
            states = states.at[batch_index, gojax.BLACK_CHANNEL_INDEX, i,
                               j].set(True)
        for i, j in [(0, 3), (1, 3), (2, 3), (3, 0), (3, 1), (3, 2)]:
            states = states.at[batch_index, gojax.WHITE_CHANNEL_INDEX, i,
                               j].set(True)

    # White has more pieces and black previously passed. White should pass to secure win.
    batch_index += 1
    for i in range(board_size // 2 + 2):
        for j in range(board_size):
            states = states.at[batch_index, gojax.WHITE_CHANNEL_INDEX, i,
                               j].set(True)
    for i in range(board_size // 2 + 2, board_size):
        for j in range(board_size):
            states = states.at[batch_index, gojax.BLACK_CHANNEL_INDEX, i,
                               j].set(True)
    # Remove the corner pieces to make it legit.
    for corner_i, corner_j in [(0, 0), (0, board_size - 1),
                               (board_size - 1, 0),
                               (board_size - 1, board_size - 1)]:
        states = states.at[batch_index, gojax.BLACK_CHANNEL_INDEX, corner_i,
                           corner_j].set(False)
        states = states.at[batch_index, gojax.WHITE_CHANNEL_INDEX, corner_i,
                           corner_j].set(False)
    states = states.at[batch_index,
                       gojax.TURN_CHANNEL_INDEX].set(gojax.WHITES_TURN)
    states = states.at[batch_index, gojax.PASS_CHANNEL_INDEX].set(True)

    return states[:batch_index + 1]


def plot_model_thoughts(go_model: hk.MultiTransformed,
                        params: optax.Params,
                        states: jnp.ndarray,
                        rng_key: jax.random.KeyArray = None):
    """
    Plots a heatmap of the policy for the given state, and bar plots of the pass and value logits.

    Plots (1) the state, (2) the non-pass action logits, (3) the pass logit.
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)
    fig, axes = plt.subplots(nrows=len(states),
                             ncols=4,
                             figsize=(12, 3 * len(states)),
                             squeeze=False)
    embeddings = go_model.apply[models.EMBED_INDEX](params, rng_key, states)
    value_logits = go_model.apply[models.VALUE_INDEX](
        params, rng_key, embeddings).astype('float32')
    policy_logits = go_model.apply[models.POLICY_INDEX](
        params, rng_key, embeddings).astype('float32')
    for i, state in enumerate(states):
        action_logits = jnp.reshape(policy_logits[i, :-1], state.shape[-2:])
        axes[i, 0].set_title('State')
        _plot_state(axes[i, 0], state)

        axes[i, 1].set_title('Action logits')
        image = axes[i, 1].imshow(action_logits)
        fig.colorbar(image, ax=axes[i, 1])

        axes[i, 2].set_title('Action probabilities')
        image = axes[i, 2].imshow(jax.nn.softmax(action_logits, axis=(0, 1)),
                                  vmin=0,
                                  vmax=1)
        fig.colorbar(image, ax=axes[i, 2])

        axes[i, 3].set_title('Pass & Value logits')
        axes[i, 3].bar(['pass', 'value'],
                       [policy_logits[i, -1], value_logits[i]])
    plt.tight_layout()


def plot_metrics(metrics_df: pd.DataFrame):
    """Plots the metrics dataframe."""
    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics_df.plot(ax=axes[0])
    metrics_df.plot(logy=True, ax=axes[1])
    plt.tight_layout()


def plot_trajectories(trajectories: data.Trajectories,
                      nt_policy_logits: jnp.ndarray = None,
                      nt_value_logits: jnp.ndarray = None):
    """Plots trajectories."""
    nrows, ncols, _, board_size, _ = trajectories.nt_states.shape
    has_logits = nt_policy_logits is not None and nt_value_logits is not None
    if has_logits:
        # State, action logits, action probabilities, pass & value logits.
        nrows *= 4
    winners = game.get_labels(trajectories.nt_states)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    for i, j in itertools.product(range(0, nrows, 4 if has_logits else 1),
                                  range(ncols)):
        # Plot state
        _plot_state(axes[i, j], trajectories.nt_states[i, j])
        # Annotate action
        action_1d = trajectories.nt_actions[i, j - 1] if j > 0 else None
        if action_1d is not None:
            if action_1d < board_size**2:
                rect = patches.Rectangle(
                    xy=(float(action_1d % board_size - 0.5),
                        float(action_1d // board_size - 0.5)),
                    width=1,
                    height=1,
                    linewidth=2,
                    edgecolor='g',
                    facecolor='none')
                axes[i, j].add_patch(rect)
        # I forgot what this does...
        axes[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[i, j].yaxis.set_major_locator(MaxNLocator(integer=True))
        # Label winner in title.
        axes[i, j].set_title({
            1: 'won',
            0: 'Tie',
            -1: 'Lost'
        }[int(winners[i, j])])

        if has_logits:
            # Plot action logits.
            action_logits = jnp.reshape(nt_policy_logits[i // 4, j, :-1],
                                        (board_size, board_size))
            axes[i + 1, j].set_title('Action logits')
            image = axes[i + 1, j].imshow(action_logits)
            fig.colorbar(image, ax=axes[i + 1, j])
            # Plot action probabilities.
            axes[i + 2, j].set_title('Action probabilities')
            image = axes[i + 2, j].imshow(jax.nn.softmax(action_logits,
                                                         axis=(0, 1)),
                                          vmin=0,
                                          vmax=1)
            fig.colorbar(image, ax=axes[i + 2, j])
            # Plot pass and value logits.
            axes[i + 3, j].set_title('Pass & Value logits')
            axes[i + 3, j].bar(
                ['pass', 'value'],
                [nt_policy_logits[i // 4, j, -1], nt_value_logits[i // 4, j]])

    plt.tight_layout()


def plot_sample_trajectories(empty_trajectories: data.Trajectories,
                             go_model: hk.MultiTransformed,
                             params: optax.Params):
    """Plots a sample of trajectories."""
    sample_traj = game.self_play(empty_trajectories, go_model, params,
                                 jax.random.PRNGKey(42))
    rng_key = jax.random.PRNGKey(42)
    states = nt_utils.flatten_first_two_dims(sample_traj.nt_states)
    embeddings = go_model.apply[models.EMBED_INDEX](params, rng_key, states)
    value_logits = go_model.apply[models.VALUE_INDEX](
        params, rng_key, embeddings).astype('float32')
    policy_logits = go_model.apply[models.POLICY_INDEX](
        params, rng_key, embeddings).astype('float32')
    batch_size, traj_length = sample_traj.nt_states.shape[:2]
    nt_value_logits = nt_utils.unflatten_first_dim(value_logits, batch_size,
                                                   traj_length)
    nt_policy_logits = nt_utils.unflatten_first_dim(policy_logits, batch_size,
                                                    traj_length)
    plot_trajectories(sample_traj, nt_policy_logits, nt_value_logits)
