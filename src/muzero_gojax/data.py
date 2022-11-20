"""Data classes."""

import chex
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class Trajectories:
    """A series of Go states and actions."""
    # [N, T, C, B, B] boolean tensor.
    nt_states: jnp.ndarray = None
    # [N, T] integer tensor.
    nt_actions: jnp.ndarray = None


@chex.dataclass(frozen=True)
class LossMetrics:
    """Loss metrics for the model."""
    decode_loss: jnp.ndarray
    decode_acc: jnp.ndarray
    value_loss: jnp.ndarray
    value_acc: jnp.ndarray
    policy_loss: jnp.ndarray
    policy_acc: jnp.ndarray
    policy_entropy: jnp.ndarray
    hypo_decode_loss: jnp.ndarray
    hypo_decode_acc: jnp.ndarray
    hypo_value_loss: jnp.ndarray
    hypo_value_acc: jnp.ndarray
    black_wins: jnp.ndarray
    ties: jnp.ndarray
    white_wins: jnp.ndarray


def init_loss_metrics(dtype: str) -> LossMetrics:
    """Initializes the train metrics with zeros with the dtype."""
    return LossMetrics(
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
        black_wins=-jnp.ones((), dtype=dtype),
        ties=-jnp.ones((), dtype=dtype),
        white_wins=-jnp.ones((), dtype=dtype),
    )
