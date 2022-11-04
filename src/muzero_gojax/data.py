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
    loss: jnp.ndarray
    acc: jnp.ndarray
    entropy: jnp.ndarray = None
    steps: jnp.ndarray = jnp.zeros((), dtype='uint8')

    def __repr__(self) -> str:
        if isinstance(self.loss, str):
            return f'Metrics[loss={self.loss}, acc={self.acc}, entropy={self.entropy}]'
        entropy_str = ''
        if isinstance(self.entropy, jnp.ndarray):
            entropy_str = f', entropy={self.entropy.item()}'
        return (f'[loss={self.loss.item()}, acc={self.acc.item()}' +
                entropy_str + ']')

    def __add__(self, other):
        entropy = None
        if isinstance(self.entropy, jnp.ndarray) and isinstance(
                other.entropy, jnp.ndarray):
            entropy = self.entropy + other.entropy
        return Metrics(loss=self.loss + other.loss,
                       acc=self.acc + other.acc,
                       entropy=entropy,
                       steps=self.steps + other.steps)

    def average(self):
        """Averages the metrics over the steps and resets the steps to 1."""
        if isinstance(self.entropy, jnp.ndarray):
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
    value: Metrics
    policy: Metrics
    trans: Metrics
    decode: Metrics

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
        kwargs: Mapping[str, Metrics] = dataclasses.asdict(self)
        return TrainMetrics(**dict(
            map(lambda item: (item[0], Metrics(**item[1]).average()),
                kwargs.items())))


def init_train_metrics(dtype: str) -> TrainMetrics:
    """Initializes the train metrics with zeros with the dtype."""
    return TrainMetrics(
        value=Metrics(loss=jnp.zeros((), dtype=dtype),
                      acc=jnp.zeros((), dtype=dtype)),
        policy=Metrics(loss=jnp.zeros((), dtype=dtype),
                       acc=jnp.zeros((), dtype=dtype),
                       entropy=jnp.zeros((), dtype=dtype)),
        trans=Metrics(loss=jnp.zeros((), dtype=dtype),
                      acc=jnp.zeros((), dtype=dtype)),
        decode=Metrics(loss=jnp.zeros((), dtype=dtype),
                       acc=jnp.zeros((), dtype=dtype)),
    )


@chex.dataclass(frozen=True)
class LossData:
    """Tracking data for computing the losses."""
    trajectories: Trajectories = None
    nt_curr_embeds: jnp.ndarray = None
    nt_original_embeds: jnp.ndarray = None
    nt_sampled_actions: jnp.ndarray = None
    nt_transition_logits: jnp.ndarray = None
    nt_game_winners: jnp.ndarray = None

    train_metrics: TrainMetrics = init_train_metrics('bfloat16')
