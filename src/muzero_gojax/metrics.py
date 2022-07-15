import re

import absl.flags
import gojax
import haiku as hk
import jax.random
import optax
import pandas as pd
from jax import numpy as jnp
from matplotlib import pyplot as plt

from muzero_gojax import game


def play_against_model(go_model: hk.MultiTransformed, params: optax.Params,
                       absl_flags: absl.flags.FlagValues):
    """
    Deploys an interactive terminal to play against the Go model.

    :param go_model: Haiku Go model.
    :param params: Model parameters.
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
        re_match = re.match('\s*(\d+)\s+(\D+)\s*', input('Enter move (R C):'))
        while not re_match:
            re_match = re.match('\s*(\d+)\s+(\D+)\s*', input('Enter move (R C):'))
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
        states = game.sample_next_states(go_model, params, rng_key, states)
        print(gojax.get_pretty_string(states[0]))
        step += 1


def plot_policy_heat_map(go_model: hk.MultiTransformed, params: optax.Params, state: jnp.ndarray,
                         rng_key: jax.random.KeyArray = None):
    """
    Plots a heatmap of the policy for the given state.

    Plots (1) the state, (2) the non-pass action logits, (3) the pass logit.
    """
    if not rng_key:
        rng_key = jax.random.PRNGKey(42)
    logits = game.get_policy_logits(go_model, params, jnp.expand_dims(state, 0), rng_key)
    action_logits, pass_logit = logits[0, :-1], logits[0, -1]
    action_logits = jnp.reshape(action_logits, state.shape[1:])
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title('State')
    plt.imshow(
        state[gojax.BLACK_CHANNEL_INDEX].astype(int) - state[gojax.WHITE_CHANNEL_INDEX].astype(int),
        vmin=-1, vmax=1, cmap='Greys')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title('Action logits')
    plt.imshow(action_logits, vmin=-3, vmax=3)
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title('Pass logit')
    plt.bar([0], [pass_logit])
    plt.ylim(-3, 3)
    plt.show()


def plot_metrics(metrics_df: pd.DataFrame):
    """Plots the metrics dataframe."""
    metrics_df.plot()
    plt.show()
