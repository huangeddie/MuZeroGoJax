"""Loss functions."""

from typing import Callable, Tuple

import chex
import gojax
import haiku as hk
import jax.nn
import jax.tree_util
import optax
from absl import flags
from jax import lax
from jax import numpy as jnp

from muzero_gojax import game, models, nt_utils

_QCOMPLETE_TEMP = flags.DEFINE_float(
    "qcomplete_temp", 0.1,
    "Temperature for q complete component policy cross entropy loss labels.")
_POLICY_TEMP = flags.DEFINE_float(
    "policy_temp", 0.1,
    "Temperature for policy logits in policy cross entropy loss labels.")
_SAMPLE_ACTION_SIZE = flags.DEFINE_integer(
    'sample_action_size', 2,
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
_WEIGHTED_VALUE_LOSS = flags.DEFINE_bool(
    "weighted_value_loss", False,
    "Whether or not weight the value losses closer towards "
    "the end of the trajectory.")
_ADD_POLICY_LOSS = flags.DEFINE_bool(
    "add_policy_loss", True,
    "Whether or not to add the policy loss to the total loss.")


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
    avg_game_length: jnp.ndarray


def _inference_nt_data(model: Callable, nt_data: jnp.ndarray,
                       **kwargs) -> jnp.ndarray:
    batch_size, traj_len = nt_data.shape[:2]
    return nt_utils.unflatten_first_dim(
        model(nt_utils.flatten_first_two_dims(nt_data), **kwargs), batch_size,
        traj_len)


def _compute_decode_metrics(
        decoded_states_logits: jnp.ndarray, trajectories: game.Trajectories,
        nt_mask: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    decode_loss = nt_utils.nt_sigmoid_cross_entropy(
        decoded_states_logits,
        trajectories.nt_states.astype(decoded_states_logits.dtype), nt_mask)
    decode_acc = jnp.nan_to_num(
        nt_utils.nt_sign_acc(decoded_states_logits,
                             trajectories.nt_states * 2 - 1, nt_mask))
    return decode_loss, decode_acc


def _compute_value_metrics(
        nt_value_logits: jnp.ndarray, nt_player_labels: jnp.ndarray,
        nt_mask: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    sigmoid_labels = (nt_player_labels + 1) / jnp.array(
        2, dtype=nt_value_logits.dtype)
    if _WEIGHTED_VALUE_LOSS.value:
        val_loss = nt_utils.nt_sigmoid_cross_entropy_linear_weights(
            nt_value_logits, labels=sigmoid_labels, nt_mask=nt_mask)
    else:
        val_loss = nt_utils.nt_sigmoid_cross_entropy(nt_value_logits,
                                                     labels=sigmoid_labels,
                                                     nt_mask=nt_mask)
    val_acc = jnp.nan_to_num(
        nt_utils.nt_sign_acc(nt_value_logits, nt_player_labels, nt_mask))
    return val_loss, val_acc


def _compute_nt_qcomplete(nt_partial_transition_value_logits: jnp.ndarray,
                          nt_value_logits: jnp.ndarray,
                          nt_sampled_actions: jnp.ndarray,
                          action_size: jnp.ndarray) -> jnp.ndarray:
    """Computes completedQ from the Gumbel Zero Paper.

    Args:
        nt_partial_transition_value_logits (jnp.ndarray): N x T x A'
        nt_value_logits (jnp.ndarray): N x T
        nt_sampled_actions (jnp.ndarray): N x T x A'
        action_size (jnp.ndarray): integer containing A.

    Returns:
        jnp.ndarray: N x T x A
    """
    # N x T x A'
    # We take the negative of the partial transitions because it's from the
    # perspective of the opponent.
    nt_partial_qvals = -nt_partial_transition_value_logits
    # N x T x A
    naive_qvals = jnp.repeat(jnp.expand_dims(nt_value_logits, axis=2),
                             repeats=action_size,
                             axis=2)
    # (N*T) x A
    flattened_naive_qvals = nt_utils.flatten_first_two_dims(naive_qvals)
    # (N*T) x A'
    flattened_sampled_actions = nt_utils.flatten_first_two_dims(
        nt_sampled_actions)
    # (N*T) x A'
    flattened_partial_qvals = nt_utils.flatten_first_two_dims(nt_partial_qvals)

    flattened_qcomplete = flattened_naive_qvals.at[
        jnp.expand_dims(jnp.arange(len(flattened_naive_qvals)), 1),
        flattened_sampled_actions].set(flattened_partial_qvals)
    batch_size, traj_len = nt_partial_transition_value_logits.shape[:2]
    return nt_utils.unflatten_first_dim(flattened_qcomplete, batch_size,
                                        traj_len)


def _compute_policy_metrics(
    policy_logits: jnp.ndarray, qcomplete: jnp.ndarray,
    nt_suffix_mask: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Updates the policy loss."""
    labels = lax.stop_gradient(qcomplete / _QCOMPLETE_TEMP.value +
                               policy_logits / _POLICY_TEMP.value)
    policy_loss = nt_utils.nt_categorical_cross_entropy(policy_logits,
                                                        labels,
                                                        nt_mask=nt_suffix_mask)
    policy_acc = nt_utils.nt_mask_mean(
        jnp.equal(jnp.argmax(policy_logits, axis=2), jnp.argmax(labels,
                                                                axis=2)),
        nt_suffix_mask).astype(policy_loss.dtype)
    policy_entropy = nt_utils.nt_entropy(policy_logits)
    return policy_loss, policy_acc, policy_entropy,


def _get_next_hypo_embed_logits(
        nt_partial_transitions: jnp.ndarray, nt_actions: jnp.ndarray,
        nt_sampled_actions: jnp.ndarray) -> jnp.ndarray:
    """Indexes the next hypothetical embeddings from the transitions.

    Args:
        nt_partial_transitions (jnp.ndarray): N x T x A' x D x B x B
        nt_actions (jnp.ndarray): N x T
        nt_sampled_actions (jnp.ndarray): N x T x A'

    Returns:
        jnp.ndarray: N x T x D x B x B
    """
    batch_size, total_steps = nt_partial_transitions.shape[:2]
    taken_action_indices = jnp.argmax(jnp.expand_dims(
        nt_actions, axis=-1) == nt_sampled_actions,
                                      axis=-1)
    flat_transitions = nt_utils.flatten_first_two_dims(nt_partial_transitions)
    # taken_transitions: (N * T) x (D*)
    taken_transitions = flat_transitions[jnp.arange(taken_action_indices.size),
                                         taken_action_indices.flatten()]
    nt_hypo_embed_logits = jnp.roll(nt_utils.unflatten_first_dim(
        taken_transitions, batch_size, total_steps),
                                    shift=1,
                                    axis=1)
    return nt_hypo_embed_logits


def _compute_loss_metrics(go_model: hk.MultiTransformed,
                          params: optax.Params,
                          trajectories: game.Trajectories,
                          rng_key=jax.random.KeyArray) -> LossMetrics:
    """
    Computes the value, and policy k-step losses.

    :param go_model: Haiku model architecture.
    :param params: Parameters of the model.
    :param trajectories: An N x T X C X H x W boolean array.
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
    nt_states = trajectories.nt_states
    nt_actions = trajectories.nt_actions
    action_size = nt_states.shape[-2] * nt_states.shape[-1] + 1
    batch_size, traj_len = nt_states.shape[:2]

    # Make masks.
    step_index = 0
    nt_suffix_mask = nt_utils.make_suffix_nt_mask(batch_size, traj_len,
                                                  traj_len - step_index)
    nt_suffix_minus_one_mask = nt_utils.make_suffix_nt_mask(
        batch_size, traj_len, traj_len - step_index - 1)

    # Get all submodel outputs.
    # Embeddings
    # N x T x D x B x B
    nt_embeds = _inference_nt_data(embed_model, nt_states)
    chex.assert_rank(nt_embeds, 5)
    # Decoded logits
    # N x T x C x B x B
    nt_decoded_states_logits = _inference_nt_data(decode_model, nt_embeds)
    # Policy logits
    # N x T x A
    nt_policy_logits = _inference_nt_data(policy_model, nt_embeds)
    chex.assert_rank(nt_policy_logits, 3)
    # Sample actions that at least include the taken action.
    # N x T x A
    indic_action_taken = jnp.reshape(
        jnp.zeros((batch_size * traj_len, action_size),
                  dtype=nt_policy_logits.dtype).at[jnp.arange(nt_actions.size),
                                                   nt_actions.flatten()].set(
                                                       float('inf')),
        nt_policy_logits.shape)
    gumbel = jax.random.gumbel(rng_key,
                               shape=nt_policy_logits.shape,
                               dtype=nt_policy_logits.dtype)
    _, nt_sampled_actions = jax.lax.top_k(nt_policy_logits + gumbel +
                                          indic_action_taken,
                                          k=_SAMPLE_ACTION_SIZE.value)
    chex.assert_equal_rank([nt_policy_logits, nt_sampled_actions])
    # N x T x A' x D x B x B
    nt_partial_transitions = _inference_nt_data(
        transition_model,
        nt_embeds,
        batch_partial_actions=nt_utils.flatten_first_two_dims(
            nt_sampled_actions))
    chex.assert_rank(nt_partial_transitions, 6)
    # N x T
    nt_value_logits = _inference_nt_data(value_model, nt_embeds)
    chex.assert_rank(nt_value_logits, 2)
    # N x T x A'
    nt_partial_transition_value_logits = nt_utils.unflatten_first_dim(
        value_model(nt_utils.flatten_first_n_dim(nt_partial_transitions, 3)),
        batch_size, traj_len, _SAMPLE_ACTION_SIZE.value)
    chex.assert_rank(nt_partial_transition_value_logits, 3)
    nt_qcomplete = _compute_nt_qcomplete(nt_partial_transition_value_logits,
                                         nt_value_logits, nt_sampled_actions,
                                         action_size)
    # Hypothetical embeddings.
    # N x T x D x B x B
    nt_hypo_embeds = _get_next_hypo_embed_logits(nt_partial_transitions,
                                                 nt_actions,
                                                 nt_sampled_actions)
    nt_hypo_value_logits = _inference_nt_data(value_model, nt_hypo_embeds)
    # Hypothetical decoded logits
    # N x T x C x B x B
    nt_hypo_decoded_states_logits = _inference_nt_data(decode_model,
                                                       nt_hypo_embeds)

    # Compute loss metrics
    decode_loss, decode_acc = _compute_decode_metrics(nt_decoded_states_logits,
                                                      trajectories,
                                                      nt_suffix_mask)
    nt_player_labels = game.get_nt_player_labels(nt_states)
    value_loss, value_acc = _compute_value_metrics(nt_value_logits,
                                                   nt_player_labels,
                                                   nt_suffix_mask)
    policy_loss, policy_acc, policy_entropy = _compute_policy_metrics(
        nt_policy_logits, nt_qcomplete, nt_suffix_mask)
    # Hypothetical embeddings invalidate the first valid indices,
    # so our suffix mask is one less.
    # TODO: Maybe compute transition metrics? pylint: disable=fixme
    hypo_value_loss, hypo_value_acc = _compute_value_metrics(
        nt_hypo_value_logits, nt_player_labels, nt_suffix_minus_one_mask)
    hypo_decode_loss, hypo_decode_acc = _compute_decode_metrics(
        nt_hypo_decoded_states_logits, trajectories, nt_suffix_minus_one_mask)
    black_wins, ties, white_wins = game.count_wins(nt_states)
    game_ended = gojax.get_ended(nt_utils.flatten_first_two_dims(nt_states))
    avg_game_length = jnp.sum(~game_ended) / batch_size
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
        black_wins=black_wins,
        ties=ties,
        white_wins=white_wins,
        avg_game_length=avg_game_length,
    )


def _extract_total_loss(
        go_model: hk.MultiTransformed, params: optax.Params,
        trajectories: game.Trajectories,
        rng_key: jax.random.KeyArray) -> Tuple[jnp.ndarray, LossMetrics]:
    """
    Computes the sum of all losses.

    Use this function to compute the gradient of the model parameters.

    :param go_model: Haiku model architecture.
    :param params: Parameters of the model.
    :param trajectories: An N x T X C X H x W boolean array.
    :return: The total loss, and metrics.
    """
    loss_metrics: LossMetrics = _compute_loss_metrics(go_model, params,
                                                      trajectories, rng_key)
    total_loss = jnp.zeros((), dtype=loss_metrics.value_loss.dtype)
    if _ADD_DECODE_LOSS.value:
        total_loss += loss_metrics.decode_loss
    if _ADD_VALUE_LOSS.value:
        total_loss += loss_metrics.value_loss
    if _ADD_POLICY_LOSS.value:
        total_loss += loss_metrics.policy_loss
    if _ADD_HYPO_VALUE_LOSS.value:
        total_loss += loss_metrics.hypo_value_loss
    if _ADD_HYPO_DECODE_LOSS.value:
        total_loss += loss_metrics.hypo_decode_loss
    return total_loss, loss_metrics


def compute_loss_gradients_and_metrics(
        go_model: hk.MultiTransformed, params: optax.Params,
        trajectories: game.Trajectories,
        rng_key: jax.random.KeyArray) -> Tuple[optax.Params, LossMetrics]:
    """Computes the gradients of the loss function."""
    loss_fn = jax.value_and_grad(_extract_total_loss, argnums=1, has_aux=True)
    (_, loss_metrics), grads = loss_fn(go_model, params, trajectories, rng_key)
    return grads, loss_metrics
