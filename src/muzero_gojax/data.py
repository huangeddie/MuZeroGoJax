"""Data classes."""

import dataclasses

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
class WinRates:
    """Player winrates"""
    black_winrate: jnp.ndarray
    white_winrate: jnp.ndarray
    tie_rate: jnp.ndarray


@chex.dataclass(frozen=True)
class SummedMetrics:
    """Loss and accuracy."""
    loss: jnp.ndarray
    acc: jnp.ndarray
    entropy: jnp.ndarray = None
    steps: jnp.ndarray = jnp.zeros((), dtype='uint8')

    def __add__(self, other):
        entropy = None
        if isinstance(self.entropy, jnp.ndarray) and isinstance(
                other.entropy, jnp.ndarray):
            entropy = self.entropy + other.entropy
        return SummedMetrics(loss=self.loss + other.loss,
                             acc=self.acc + other.acc,
                             entropy=entropy,
                             steps=self.steps + other.steps)

    def average(self):
        """Averages the metrics over the steps and resets the steps to 1."""
        if isinstance(self.entropy, jnp.ndarray):
            entropy = self.entropy / self.steps
        else:
            entropy = None
        return SummedMetrics(loss=self.loss / self.steps,
                             acc=self.acc / self.steps,
                             entropy=entropy,
                             steps=jnp.ones((), dtype='uint8'))


@chex.dataclass(frozen=True)
class TrainMetrics:
    """Training metrics."""
    value: SummedMetrics
    policy: SummedMetrics
    trans: SummedMetrics
    decode: SummedMetrics
    win_rates: WinRates

    def update_decode(self, other_decode):
        #pylint: disable=missing-function-docstring
        return dataclasses.replace(self, decode=self.decode + other_decode)

    def update_value(self, other_value):
        #pylint: disable=missing-function-docstring
        return dataclasses.replace(self, value=self.value + other_value)

    def update_policy(self, other_policy):
        #pylint: disable=missing-function-docstring
        return dataclasses.replace(self, policy=self.policy + other_policy)

    def update_trans(self, other_trans):
        #pylint: disable=missing-function-docstring
        return dataclasses.replace(self, trans=self.trans + other_trans)

    def average(self):
        """Averages the metrics over the steps and resets the steps to 1."""
        return TrainMetrics(value=self.value.average(),
                            policy=self.policy.average(),
                            trans=self.trans.average(),
                            decode=self.decode.average(),
                            win_rates=self.win_rates)


def init_train_metrics(dtype: str) -> TrainMetrics:
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
    )


@chex.dataclass(frozen=True)
class LossData:
    """Tracking data for computing the losses."""
    trajectories: Trajectories
    nt_curr_embeds: jnp.ndarray
    nt_original_embeds: jnp.ndarray
    nt_sampled_actions: jnp.ndarray
    nt_transition_logits: jnp.ndarray
    nt_player_labels: jnp.ndarray
    train_metrics: TrainMetrics


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
