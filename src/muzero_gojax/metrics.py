"""Module for understanding the behavior of the code."""
import itertools

import gojax
import haiku as hk
import jax.random
import optax
import pandas as pd
from jax import numpy as jnp
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from muzero_gojax import game, models, nt_utils


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

    # Black holds 4x4 top left corner, holds more pieces, and previously passed.
    # Black should pass to secure win.
    batch_index += 1
    for i, j in zip(range(3, -1, -1), range(4)):
        states = states.at[batch_index, gojax.BLACK_CHANNEL_INDEX, i,
                           j].set(True)
    states = states.at[batch_index, gojax.BLACK_CHANNEL_INDEX, 1, 0].set(True)
    states = states.at[batch_index, gojax.WHITE_CHANNEL_INDEX, board_size - 1,
                       board_size - 1].set(True)
    states = states.at[batch_index, gojax.PASS_CHANNEL_INDEX].set(True)

    # White holds 4x4 top left corner, holds more pieces.
    # Model should know it's losing.
    batch_index += 1
    for i, j in zip(range(3, -1, -1), range(4)):
        states = states.at[batch_index, gojax.WHITE_CHANNEL_INDEX, i,
                           j].set(True)
    states = states.at[batch_index, gojax.WHITE_CHANNEL_INDEX, 1, 0].set(True)
    states = states.at[batch_index, gojax.BLACK_CHANNEL_INDEX, board_size - 1,
                       board_size - 1].set(True)

    # Black is surrounded but has two holes. It should not fill either hole.
    if board_size >= 4:
        batch_index += 1
        for i, j in [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]:
            states = states.at[batch_index, gojax.BLACK_CHANNEL_INDEX, i,
                               j].set(True)
        for i, j in [(0, 3), (1, 3), (2, 3), (3, 0), (3, 1), (3, 2)]:
            states = states.at[batch_index, gojax.WHITE_CHANNEL_INDEX, i,
                               j].set(True)

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


def plot_train_metrics_by_regex(train_metrics_df: pd.DataFrame, regexes=None):
    """Plots the metrics dataframe grouped by regex's."""
    if regexes is None:
        regexes = [
            '.+_entropy',
            '.+_acc',
            '.+_loss',
            '(.+_wins|ties|avg_game_length)',
        ]
    _, axes = plt.subplots(len(regexes),
                           2,
                           figsize=(12, 3 * len(regexes)),
                           squeeze=False)
    for i, regex in enumerate(regexes):
        sub_df = train_metrics_df.filter(regex=regex)
        sub_df.plot(ax=axes[i, 0])
        sub_df.plot(logy=True, ax=axes[i, 1])
    plt.tight_layout()


def plot_trajectories(trajectories: game.Trajectories,
                      nt_policy_logits: jnp.ndarray = None,
                      nt_value_logits: jnp.ndarray = None):
    """Plots trajectories."""
    batch_size, traj_len, _, board_size, _ = trajectories.nt_states.shape
    has_logits = nt_policy_logits is not None and nt_value_logits is not None
    if has_logits:
        # State, action logits, action probabilities, pass & value logits.
        nrows = batch_size * 4
    else:
        nrows = batch_size

    player_labels = game.get_nt_player_labels(trajectories.nt_states)
    fig, axes = plt.subplots(nrows,
                             traj_len,
                             figsize=(traj_len * 3, nrows * 3))
    for batch_idx, traj_idx in itertools.product(range(batch_size),
                                                 range(traj_len)):
        if has_logits:
            group_start_row_idx = batch_idx * 4
        else:
            group_start_row_idx = batch_idx
        # Plot state
        _plot_state(axes[group_start_row_idx, traj_idx],
                    trajectories.nt_states[batch_idx, traj_idx])
        # Annotate action
        action_1d = trajectories.nt_actions[batch_idx, traj_idx -
                                            1] if traj_idx > 0 else None
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
                axes[group_start_row_idx, traj_idx].add_patch(rect)
        # I forgot what this does...
        axes[group_start_row_idx,
             traj_idx].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[group_start_row_idx,
             traj_idx].yaxis.set_major_locator(MaxNLocator(integer=True))
        # Label winner in title.
        axes[group_start_row_idx, traj_idx].set_title({
            1: 'won',
            0: 'Tie',
            -1: 'Lost'
        }[int(player_labels[batch_idx, traj_idx])])

        if has_logits:
            # Plot action logits.
            action_logits = jnp.reshape(
                nt_policy_logits[batch_idx, traj_idx, :-1],
                (board_size, board_size))
            axes[group_start_row_idx + 1, traj_idx].set_title('Action logits')
            image = axes[group_start_row_idx + 1,
                         traj_idx].imshow(action_logits)
            fig.colorbar(image, ax=axes[group_start_row_idx + 1, traj_idx])
            # Plot action probabilities.
            axes[group_start_row_idx + 2,
                 traj_idx].set_title('Action probabilities')
            image = axes[group_start_row_idx + 2,
                         traj_idx].imshow(jax.nn.softmax(action_logits,
                                                         axis=(0, 1)),
                                          vmin=0,
                                          vmax=1)
            fig.colorbar(image, ax=axes[group_start_row_idx + 2, traj_idx])
            # Plot pass and value logits.
            axes[group_start_row_idx + 3,
                 traj_idx].set_title('Pass & Value logits')
            axes[group_start_row_idx + 3, traj_idx].bar(['pass', 'value'], [
                nt_policy_logits[batch_idx, traj_idx, -1],
                nt_value_logits[batch_idx, traj_idx]
            ])

    plt.tight_layout()


def plot_sample_trajectories(empty_trajectories: game.Trajectories,
                             go_model: hk.MultiTransformed,
                             params: optax.Params):
    """Plots a sample of trajectories."""
    policy_model = models.get_policy_model(go_model, params)
    sample_traj = game.self_play(empty_trajectories, policy_model,
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
