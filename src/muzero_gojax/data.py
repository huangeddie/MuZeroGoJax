"""Processes and samples data from games for model updates."""

import chex
import gojax
import jax
import jax.numpy as jnp

from muzero_gojax import game, nt_utils


@chex.dataclass(frozen=True)
class GameData:
    """Game data.

    `k` represents 1 + the number of hypothetical steps. 
    By default, k=2 since the number of hypothetical steps is 1 by default.
    """
    nk_states: jnp.ndarray
    nk_actions: jnp.ndarray
    nk_player_labels: jnp.ndarray  # {-1, 0, 1}


def sample_game_data(trajectories: game.Trajectories,
                     rng_key: jax.random.KeyArray) -> GameData:
    """Samples game data from trajectories."""
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
    return GameData(nk_states=nk_states,
                    nk_actions=nk_actions,
                    nk_player_labels=nk_player_labels)
