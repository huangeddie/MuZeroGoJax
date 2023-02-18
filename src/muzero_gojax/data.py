"""Processes and samples data from games for model updates."""

import chex
import gojax
import jax
import jax.numpy as jnp

from muzero_gojax import game, nt_utils


@chex.dataclass(frozen=True)
class GameData:
    """Game data.

    The model is trained to predict the following:
    • The end state given the start state and the actions taken
    • The end reward given the start state and the actions taken
    • The start reward given the start state
    """
    start_states: jnp.ndarray
    # Actions taken from start state to end state. A value of -1 indicates that
    # the previous value was the last action taken. k is currently hardcoded to
    # 5 because we assume the max number of hypothetical steps we'll
    # use is 4.
    nk_actions: jnp.ndarray
    end_states: jnp.ndarray
    # TODO: Rename to start_player_labels
    start_labels: jnp.ndarray  # {-1, 0, 1}
    # TODO: Rename to end_player_labels
    end_labels: jnp.ndarray  # {-1, 0, 1}
    # TODO: Add sampled q-values.


def sample_game_data(trajectories: game.Trajectories,
                     rng_key: jax.random.KeyArray,
                     max_hypo_steps: int) -> GameData:
    """Samples game data from trajectories.

    For each trajectory, we independently sample our hypothetical step value k
    uniformly from [1, max_hypo_steps]. We then sample two states from
    the trajectory. The index of the first state, i, is sampled uniformly from
    all non-terminal states of the trajectory. The index of the second state, j,
    is min(i+k, n) where n is the length of the trajectory 
    (including the terminal state).

    Args:
        trajectories: Trajectories from a game.
        rng_key: Random key for sampling.
        max_hypo_steps: Maximum number of hypothetical steps to use.

    Returns:
        Game data sampled from trajectories.
    """
    max_max_hypo_steps = 5
    if max_hypo_steps >= max_max_hypo_steps:
        raise ValueError(f'max_hypo_steps must be < {max_max_hypo_steps}.')
    batch_size, traj_len = trajectories.nt_states.shape[:2]
    next_k_indices = jnp.repeat(jnp.expand_dims(jnp.arange(max_max_hypo_steps),
                                                axis=0),
                                batch_size,
                                axis=0)
    batch_order_indices = jnp.arange(batch_size)
    game_ended = nt_utils.unflatten_first_dim(
        gojax.get_ended(nt_utils.flatten_first_two_dims(
            trajectories.nt_states)), batch_size, traj_len)
    base_sample_state_logits = game_ended * float('-inf')
    game_len = jnp.sum(~game_ended, axis=1)
    start_indices = jax.random.categorical(rng_key,
                                           base_sample_state_logits,
                                           axis=1)
    chex.assert_rank(start_indices, 1)
    unclamped_hypo_steps = jax.random.randint(rng_key,
                                              shape=(batch_size, ),
                                              minval=1,
                                              maxval=max_hypo_steps + 1)
    chex.assert_equal_shape([start_indices, game_len, unclamped_hypo_steps])
    end_indices = jnp.minimum(start_indices + unclamped_hypo_steps, game_len)
    hypo_steps = jnp.minimum(unclamped_hypo_steps, end_indices - start_indices)
    start_states = trajectories.nt_states[batch_order_indices, start_indices]
    end_states = trajectories.nt_states[batch_order_indices, end_indices]
    # TODO: Use int16 instead of int32.
    nk_actions = trajectories.nt_actions[
        jnp.expand_dims(batch_order_indices, axis=1),
        jnp.expand_dims(start_indices, axis=1) +
        next_k_indices].astype('int32')
    chex.assert_shape(nk_actions, (batch_size, max_max_hypo_steps))
    nk_actions = jnp.where(
        next_k_indices < jnp.expand_dims(hypo_steps, axis=1), nk_actions,
        jnp.full_like(nk_actions, fill_value=-1))
    nt_player_labels = game.get_nt_player_labels(trajectories.nt_states)
    start_labels = nt_player_labels[batch_order_indices, start_indices]
    end_labels = nt_player_labels[batch_order_indices, end_indices]
    return GameData(start_states=start_states,
                    end_states=end_states,
                    nk_actions=nk_actions,
                    start_labels=start_labels,
                    end_labels=end_labels)
