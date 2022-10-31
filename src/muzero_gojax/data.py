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

    def __repr__(self) -> str:
        if self.loss is str:
            return super().__repr__()
        entropy_str = ''
        if self.entropy is not None:
            entropy_str = f', entropy={self.entropy.item()}'
        return (f'[loss={self.loss.item()}, acc={self.acc.item()}' +
                entropy_str + ']')

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
    value: Metrics = Metrics()
    policy: Metrics = Metrics(entropy=jnp.zeros((), dtype='bfloat16'))
    trans: Metrics = Metrics()
    decode: Metrics = Metrics()

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
