"""Data classes."""

import dataclasses
from typing import Mapping

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
class Metrics:
    """Loss and accuracy."""
    loss: jnp.ndarray = jnp.zeros((), dtype='bfloat16')
    acc: jnp.ndarray = jnp.zeros((), dtype='bfloat16')
    entropy: jnp.ndarray = None
    steps: jnp.ndarray = jnp.zeros((), dtype='uint8')

    def __str__(self) -> str:
        entropy = ''
        if self.entropy is not None:
            entropy = self.entropy.item()
        return (f'(loss={self.loss.item()}, '
                f'acc={self.acc.item()}, '
                f'entropy={entropy})')

    def __add__(self, other):
        entropy = None
        if self.entropy is not None and other.entropy is not None:
            entropy = self.entropy + other.entropy
        return Metrics(loss=self.loss + other.loss,
                       acc=self.acc + other.acc,
                       entropy=entropy,
                       steps=self.steps + other.steps)

    def average(self):
        """Averages the metrics over the steps and resets the steps to 1."""
        if self.entropy is not None:
            entropy = self.entropy / self.steps
        else:
            entropy = None
        return Metrics(loss=self.loss / self.steps,
                       acc=self.acc / self.steps,
                       entropy=entropy,
                       steps=jnp.ones((), dtype='uint8'))


@chex.dataclass(frozen=True)
class TrainMetrics:
    """Training metrics."""
    value_metrics: Metrics = Metrics()
    policy_metrics: Metrics = Metrics(entropy=jnp.zeros((), dtype='bfloat16'))
    trans_metrics: Metrics = Metrics()
    decode_metrics: Metrics = Metrics()

    def update_decode_metrics(self, other_decode_metrics):
        return self.replace(decode_metrics=self.decode_metrics +
                            other_decode_metrics)

    def update_value_metrics(self, other_value_metrics):
        return self.replace(value_metrics=self.value_metrics +
                            other_value_metrics)

    def update_policy_metrics(self, other_policy_metrics):
        return self.replace(policy_metrics=self.policy_metrics +
                            other_policy_metrics)

    def update_trans_metrics(self, other_trans_metrics):
        return self.replace(trans_metrics=self.trans_metrics +
                            other_trans_metrics)

    def average(self):
        """Averages the metrics over the steps and resets the steps to 1."""
        kwargs: Mapping[str, Metrics] = dataclasses.asdict(self)
        return TrainMetrics(**dict(
            map(lambda item: (item[0], Metrics(**item[1]).average()),
                kwargs.items())))


@chex.dataclass(frozen=True)
class LossData:
    """Tracking data for computing the losses."""
    trajectories: Trajectories = None
    nt_curr_embeds: jnp.ndarray = None
    nt_original_embeds: jnp.ndarray = None
    nt_sampled_actions: jnp.ndarray = None
    nt_transition_logits: jnp.ndarray = None
    nt_game_winners: jnp.ndarray = None

    train_metrics: TrainMetrics = TrainMetrics()
