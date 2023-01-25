"""Processes and samples data from games for model updates."""

import chex
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class GameData:
    """Game data.

    `k` represents 1 + the number of hypothetical steps. 
    By default, k=2 since the number of hypothetical steps is 1 by default.
    """
    nk_states: jnp.ndarray
    nk_actions: jnp.ndarray
    nk_player_labels: jnp.ndarray  # {-1, 0, 1}