"""Loss functions."""

from typing import Tuple

import chex
import haiku as hk
import jax.nn
import jax.tree_util
import optax
from absl import flags
from jax import lax
from jax import numpy as jnp

from muzero_gojax import models, nt_utils

_QCOMPLETE_TEMP = flags.DEFINE_float(
    "qcomplete_temp", 1,
    "Temperature for q complete component policy cross entropy loss labels.")
_POLICY_TEMP = flags.DEFINE_float(
    "policy_temp", 1,
    "Temperature for policy logits in policy cross entropy loss labels.")
_LOSS_SAMPLE_ACTION_SIZE = flags.DEFINE_integer(
    'loss_sample_action_size', 2,
    'Number of actions to sample from for policy improvement.')
_ADD_DECODE_LOSS = flags.DEFINE_bool(
    "add_decode_loss", True,
    "Whether or not to add the decode loss to the total loss.")
_ADD_HYPO_DECODE_LOSS = flags.DEFINE_bool(
    "add_hypo_decode_loss", True,
    "Whether or not to add the hypothetical decode loss to the total loss.")
_ADD_VALUE_LOSS = flags.DEFINE_bool(
    "add_value_loss", True,
    "Whether or not to add the value loss to the total loss.")
_ADD_HYPO_VALUE_LOSS = flags.DEFINE_bool(
    "add_hypo_value_loss", True,
    "Whether or not to add the hypothetical value loss to the total loss.")
_ADD_POLICY_LOSS = flags.DEFINE_bool(
    "add_policy_loss", True,
    "Whether or not to add the policy loss to the total loss.")
_POLICY_LOSS_SCALE = flags.DEFINE_float("policy_loss_scale", 1,
                                        "Scale constant on the policy loss.")


@chex.dataclass(frozen=True)
class GameData:
    """Game data.

    `k` represents 1 + the number of hypothetical steps. 
    By default, k=2 since the number of hypothetical steps is 1 by default.
    """
    nk_states: jnp.ndarray
    nk_actions: jnp.ndarray
    nk_player_labels: jnp.ndarray  # {-1, 0, 1}


@chex.dataclass(frozen=True)
class LossMetrics:
    """Loss metrics for the model."""
    decode_loss: jnp.ndarray
    decode_acc: jnp.ndarray
    value_loss: jnp.ndarray
    value_acc: jnp.ndarray
    policy_loss: jnp.ndarray  # KL divergence.
    policy_acc: jnp.ndarray
    policy_entropy: jnp.ndarray
    hypo_decode_loss: jnp.ndarray
    hypo_decode_acc: jnp.ndarray
    hypo_value_loss: jnp.ndarray
    hypo_value_acc: jnp.ndarray


def _compute_decode_metrics(
        decoded_states_logits: jnp.ndarray,
        states: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    states = states.astype(decoded_states_logits.dtype)
    cross_entropy = -states * jax.nn.log_sigmoid(decoded_states_logits) - (
        1 - states) * jax.nn.log_sigmoid(-decoded_states_logits)
    decode_loss = jnp.mean(cross_entropy)
    decode_acc = jnp.mean(
        jnp.sign(decoded_states_logits) == jnp.sign(states * 2 - 1),
        dtype=decoded_states_logits.dtype)
    return decode_loss, decode_acc


def _compute_value_metrics(
        value_logits: jnp.ndarray,
        player_labels: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    sigmoid_labels = (player_labels + 1) / jnp.array(2,
                                                     dtype=value_logits.dtype)
    cross_entropy = -sigmoid_labels * jax.nn.log_sigmoid(value_logits) - (
        1 - sigmoid_labels) * jax.nn.log_sigmoid(-value_logits)
    val_loss = jnp.mean(cross_entropy)
    val_acc = jnp.mean(jnp.sign(value_logits) == jnp.sign(player_labels),
                       dtype=value_logits.dtype)
    return val_loss, val_acc


def _compute_qcomplete(partial_transition_value_logits: jnp.ndarray,
                       value_logits: jnp.ndarray, sampled_actions: jnp.ndarray,
                       action_size: jnp.ndarray) -> jnp.ndarray:
    """Computes completedQ from the Gumbel Zero Paper.

    Args:
        partial_transition_value_logits (jnp.ndarray): N x A'
        value_logits (jnp.ndarray): N
        sampled_actions (jnp.ndarray): N x A'
        action_size (jnp.ndarray): integer containing A.

    Returns:
        jnp.ndarray: N x A
    """
    chex.assert_equal_shape((partial_transition_value_logits, sampled_actions))

    # N x A'
    # We take the negative of the partial transitions because it's from the
    # perspective of the opponent.
    partial_qvals = -partial_transition_value_logits
    # N x A
    naive_qvals = jnp.repeat(jnp.expand_dims(value_logits, axis=1),
                             repeats=action_size,
                             axis=1)
    return naive_qvals.at[jnp.expand_dims(jnp.arange(len(naive_qvals)), 1),
                          sampled_actions].set(partial_qvals)


def _compute_policy_metrics(
        policy_logits: jnp.ndarray, qcomplete: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Updates the policy loss."""
    labels = lax.stop_gradient(qcomplete / _QCOMPLETE_TEMP.value +
                               policy_logits / _POLICY_TEMP.value)
    cross_entropy = -jnp.sum(
        jax.nn.softmax(labels) * jax.nn.log_softmax(policy_logits), axis=-1)
    target_entropy = -jnp.sum(
        jax.nn.softmax(labels) * jax.nn.log_softmax(labels), axis=-1)
    policy_entropy = jnp.mean(-jnp.sum(jax.nn.softmax(policy_logits) *
                                       jax.nn.log_softmax(policy_logits),
                                       axis=-1))
    policy_loss = jnp.mean(cross_entropy - target_entropy)
    policy_acc = jnp.mean(
        jnp.equal(jnp.argmax(policy_logits, axis=1),
                  jnp.argmax(labels, axis=1))).astype(policy_loss.dtype)
    return policy_loss, policy_acc, policy_entropy,


def _get_next_hypo_embed_logits(partial_transitions: jnp.ndarray,
                                actions: jnp.ndarray,
                                sampled_actions: jnp.ndarray) -> jnp.ndarray:
    """Indexes the next hypothetical embeddings from the transitions.

    Args:
        partial_transitions (jnp.ndarray): N x A' x D x B x B
        actions (jnp.ndarray): N \in [A]
        sampled_actions (jnp.ndarray): N x A'

    Returns:
        jnp.ndarray: N x D x B x B
    """
    # N
    taken_action_indices = jnp.argmax(jnp.expand_dims(
        actions, axis=1) == sampled_actions,
                                      axis=1)
    chex.assert_equal_shape((taken_action_indices, actions))
    # taken_transitions: N x D x B x B
    return partial_transitions[jnp.arange(len(taken_action_indices)),
                               taken_action_indices]


def _compute_loss_metrics(go_model: hk.MultiTransformed,
                          params: optax.Params,
                          game_data: GameData,
                          rng_key=jax.random.KeyArray) -> LossMetrics:
    """
    Computes the value, and policy k-step losses.

    :param go_model: Haiku model architecture.
    :param params: Parameters of the model.
    :param game_data: Game data.
    :return: A dictionary of cumulative losses and model state
    """

    # Get basic info.
    embed_model = jax.tree_util.Partial(go_model.apply[models.EMBED_INDEX],
                                        params, rng_key)
    decode_model = jax.tree_util.Partial(go_model.apply[models.DECODE_INDEX],
                                         params, rng_key)
    value_model = jax.tree_util.Partial(go_model.apply[models.VALUE_INDEX],
                                        params, rng_key)
    policy_model = jax.tree_util.Partial(go_model.apply[models.POLICY_INDEX],
                                         params, rng_key)
    transition_model = jax.tree_util.Partial(
        go_model.apply[models.TRANSITION_INDEX], params, rng_key)
    nk_states = game_data.nk_states
    nk_actions = game_data.nk_actions
    base_states = nk_states[:, 0]
    base_actions = nk_actions[:, 0]
    action_size = nk_states.shape[-2] * nk_states.shape[-1] + 1
    batch_size = len(nk_states)

    # Get all submodel outputs.
    # Embeddings
    # N x D x B x B
    embeds = embed_model(base_states)
    chex.assert_rank(embeds, 4)
    # Decoded logits
    # N x C x B x B
    decoded_states_logits = decode_model(embeds)
    # Policy logits
    # N x A
    policy_logits = policy_model(embeds)
    chex.assert_shape(policy_logits, (batch_size, action_size))
    # Sample actions that at least include the taken action.
    # N x A
    indic_action_taken = jnp.zeros(
        (batch_size, action_size),
        dtype=policy_logits.dtype).at[jnp.arange(len(nk_actions)),
                                      base_actions].set(float('inf'))
    chex.assert_equal_shape((policy_logits, indic_action_taken))
    gumbel = jax.random.gumbel(rng_key,
                               shape=policy_logits.shape,
                               dtype=policy_logits.dtype)
    _, sampled_actions = jax.lax.top_k(policy_logits + gumbel +
                                       indic_action_taken,
                                       k=_LOSS_SAMPLE_ACTION_SIZE.value)
    chex.assert_equal_rank([policy_logits, sampled_actions])
    # N x A' x D x B x B
    partial_transitions = transition_model(
        embeds, batch_partial_actions=sampled_actions)
    chex.assert_rank(partial_transitions, 5)
    # N
    value_logits = value_model(embeds)
    chex.assert_rank(value_logits, 1)
    # N x A'
    partial_transition_value_logits = nt_utils.unflatten_first_dim(
        value_model(nt_utils.flatten_first_two_dims(partial_transitions)),
        batch_size, _LOSS_SAMPLE_ACTION_SIZE.value)
    chex.assert_rank(partial_transition_value_logits, 2)
    qcomplete = _compute_qcomplete(partial_transition_value_logits,
                                   value_logits, sampled_actions, action_size)
    # Hypothetical embeddings.
    # N x D x B x B
    hypo_embeds = _get_next_hypo_embed_logits(partial_transitions,
                                              base_actions, sampled_actions)
    chex.assert_rank(hypo_embeds, 4)
    hypo_value_logits = value_model(hypo_embeds)
    # Hypothetical decoded logits
    # N x C x B x B
    hypo_decoded_states_logits = decode_model(hypo_embeds)

    # Compute loss metrics
    decode_loss, decode_acc = _compute_decode_metrics(decoded_states_logits,
                                                      base_states)
    nk_player_labels = game_data.nk_player_labels
    chex.assert_rank(nk_player_labels, 2)
    value_loss, value_acc = _compute_value_metrics(value_logits,
                                                   nk_player_labels[:, 0])
    policy_loss, policy_acc, policy_entropy = _compute_policy_metrics(
        policy_logits, qcomplete)
    # Hypothetical embeddings invalidate the first valid indices,
    # so our suffix mask is one less.
    # TODO: Maybe compute transition metrics? pylint: disable=fixme
    hypo_value_loss, hypo_value_acc = _compute_value_metrics(
        hypo_value_logits, nk_player_labels[:, 1])
    hypo_decode_loss, hypo_decode_acc = _compute_decode_metrics(
        hypo_decoded_states_logits, nk_states[:, 1])
    return LossMetrics(
        decode_loss=decode_loss,
        decode_acc=decode_acc,
        value_loss=value_loss,
        value_acc=value_acc,
        policy_loss=policy_loss,
        policy_acc=policy_acc,
        policy_entropy=policy_entropy,
        hypo_value_loss=hypo_value_loss,
        hypo_value_acc=hypo_value_acc,
        hypo_decode_loss=hypo_decode_loss,
        hypo_decode_acc=hypo_decode_acc,
    )


def _extract_total_loss(
        go_model: hk.MultiTransformed, params: optax.Params,
        game_data: GameData,
        rng_key: jax.random.KeyArray) -> Tuple[jnp.ndarray, LossMetrics]:
    """
    Computes the sum of all losses.

    Use this function to compute the gradient of the model parameters.

    :param go_model: Haiku model architecture.
    :param params: Parameters of the model.
    :param game_data: Game data.
    :return: The total loss, and metrics.
    """
    loss_metrics: LossMetrics = _compute_loss_metrics(go_model, params,
                                                      game_data, rng_key)
    total_loss = jnp.zeros((), dtype=loss_metrics.value_loss.dtype)
    if _ADD_DECODE_LOSS.value:
        total_loss += loss_metrics.decode_loss
    if _ADD_VALUE_LOSS.value:
        total_loss += loss_metrics.value_loss
    if _ADD_POLICY_LOSS.value:
        total_loss += _POLICY_LOSS_SCALE.value * loss_metrics.policy_loss
    if _ADD_HYPO_VALUE_LOSS.value:
        total_loss += loss_metrics.hypo_value_loss
    if _ADD_HYPO_DECODE_LOSS.value:
        total_loss += loss_metrics.hypo_decode_loss
    return total_loss, loss_metrics


def compute_loss_gradients_and_metrics(
        go_model: hk.MultiTransformed, params: optax.Params,
        game_data: GameData,
        rng_key: jax.random.KeyArray) -> Tuple[optax.Params, LossMetrics]:
    """Computes the gradients of the loss function."""
    loss_fn = jax.value_and_grad(_extract_total_loss, argnums=1, has_aux=True)
    (_, loss_metrics), grads = loss_fn(go_model, params, game_data, rng_key)
    return grads, loss_metrics
