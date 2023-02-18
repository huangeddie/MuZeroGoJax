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

from muzero_gojax import data, models, nt_utils

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
    """Computes sigmoid cross entropy loss and accuracy.

    Args:
        value_logits (jnp.ndarray): Value logits
        player_labels (jnp.ndarray): Whether the player won, target values. 
            {-1, 0, 1}

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Sigmoid cross entropy loss and accuracy.
    """
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


@chex.dataclass(frozen=True)
class TransitionData:
    """Dataclass for transition data."""
    transition_model: jax.tree_util.Partial
    current_embeddings: jnp.ndarray
    nk_actions: jnp.ndarray
    rng_key: jax.random.KeyArray


def _iterate_transitions(for_i: int,
                         transition_data: TransitionData) -> TransitionData:
    """Updates the current embeddings.

    Args:
        for_i (int): Index of the iteration and column of the nk_actions array.
        transition_data (TransitionData): Transition data.

    Returns:
        TransitionData: Updated transition data.
    """
    actions_taken = transition_data.nk_actions[:, for_i]
    chex.assert_type(actions_taken, 'int32')
    # N x D x B x B
    transitioned_embeddings = jnp.squeeze(transition_data.transition_model(
        transition_data.rng_key, transition_data.current_embeddings,
        jnp.expand_dims(actions_taken, axis=1)),
                                          axis=1)
    next_embeddings = jnp.where(
        jnp.expand_dims(actions_taken, axis=(1, 2, 3)) >= 0,
        transitioned_embeddings, transition_data.current_embeddings)
    return transition_data.replace(current_embeddings=next_embeddings,
                                   rng_key=jax.random.fold_in(
                                       transition_data.rng_key, for_i))


def _compute_loss_metrics(go_model: hk.MultiTransformed, params: optax.Params,
                          game_data: data.GameData,
                          rng_key: jax.random.KeyArray) -> LossMetrics:
    """Computes loss metrics based on model features from the game data.


    Args:
        go_model (hk.MultiTransformed): Haiku model architecture.
        params (optax.Params): Parameters of the model.
        game_data (data.GameData): Game data.
        rng_key (jax.random.KeyArray): Rng Key

    Returns:
        LossMetrics: Loss metrics.
    """

    # Compute the value metrics on the start states.

    rng_key, embed_key = jax.random.split(rng_key)
    # N x D x B x B
    start_state_embeds = go_model.apply[models.EMBED_INDEX](
        params, embed_key, game_data.start_states)
    del embed_key
    chex.assert_rank(start_state_embeds, 4)
    # N
    rng_key, value_key = jax.random.split(rng_key)
    value_logits = go_model.apply[models.VALUE_INDEX](params, value_key,
                                                      start_state_embeds)
    del value_key
    chex.assert_rank(value_logits, 1)
    value_loss, value_acc = _compute_value_metrics(value_logits,
                                                   game_data.start_labels)

    # Compute decode metrics on the start states.
    rng_key, decode_key = jax.random.split(rng_key)
    decode_start_state_logits = go_model.apply[models.DECODE_INDEX](
        params, decode_key, start_state_embeds)
    del decode_key
    chex.assert_equal_shape(
        (decode_start_state_logits, game_data.start_states))
    decode_loss, decode_acc = _compute_decode_metrics(
        decode_start_state_logits, game_data.start_states)

    # Compute the hypothetical value metrics based on transitions embeddings on
    # the end state.
    chex.assert_rank(game_data.nk_actions, 2)
    transition_model = jax.tree_util.Partial(
        go_model.apply[models.TRANSITION_INDEX], params)
    rng_key, transition_key = jax.random.split(rng_key)
    end_state_hypo_embeddings = lax.fori_loop(
        0, game_data.nk_actions.shape[1], _iterate_transitions,
        TransitionData(transition_model=transition_model,
                       current_embeddings=start_state_embeds,
                       nk_actions=game_data.nk_actions,
                       rng_key=transition_key)).current_embeddings
    del transition_key
    chex.assert_equal_shape((end_state_hypo_embeddings, start_state_embeds))
    rng_key, hypo_value_key = jax.random.split(rng_key)
    hypo_value_logits = go_model.apply[models.VALUE_INDEX](
        params, hypo_value_key, end_state_hypo_embeddings)
    del hypo_value_key
    chex.assert_equal_shape((hypo_value_logits, value_logits))
    hypo_value_loss, hypo_value_acc = _compute_value_metrics(
        hypo_value_logits, game_data.end_labels)

    # Compute the hypothetical decode metrics based on transitions embeddings
    # on the end state.
    rng_key, hypo_decode_key = jax.random.split(rng_key)
    hypo_decode_end_state_logits = go_model.apply[models.DECODE_INDEX](
        params, hypo_decode_key, end_state_hypo_embeddings)
    del hypo_decode_key
    chex.assert_equal_shape(
        (hypo_decode_end_state_logits, game_data.end_states))
    hypo_decode_loss, hypo_decode_acc = _compute_decode_metrics(
        hypo_decode_end_state_logits, game_data.end_states)

    # Compute policy metrics on the start states.
    # N x A
    rng_key, policy_key = jax.random.split(rng_key)
    policy_logits = go_model.apply[models.POLICY_INDEX](params, policy_key,
                                                        start_state_embeds)
    del policy_key
    chex.assert_rank(policy_logits, 2)
    # Use the Gumbel top-k trick to select k actions without replacement.
    # Note this is not related to the MuZero Gumbel policy improvement.
    rng_key, gumbel_key = jax.random.split(rng_key)
    gumbel = jax.random.gumbel(gumbel_key,
                               shape=policy_logits.shape,
                               dtype=policy_logits.dtype)
    del gumbel_key
    # N x A'
    _, sampled_actions = jax.lax.top_k(policy_logits + gumbel,
                                       k=_LOSS_SAMPLE_ACTION_SIZE.value)
    chex.assert_equal_rank([policy_logits, sampled_actions])
    # N x A' x D x B x B
    rng_key, partial_transition_key = jax.random.split(rng_key)
    partial_transitions = go_model.apply[models.TRANSITION_INDEX](
        params,
        partial_transition_key,
        start_state_embeds,
        batch_partial_actions=sampled_actions)
    del partial_transition_key
    chex.assert_rank(partial_transitions, 5)
    # N x A'
    batch_size = len(game_data.start_states)
    rng_key, partial_transition_value_key = jax.random.split(rng_key)
    # N x A'
    partial_transition_value_logits = nt_utils.unflatten_first_dim(
        go_model.apply[models.VALUE_INDEX](
            params, partial_transition_value_key,
            nt_utils.flatten_first_two_dims(partial_transitions)), batch_size,
        _LOSS_SAMPLE_ACTION_SIZE.value)
    del partial_transition_value_key
    chex.assert_rank(partial_transition_value_logits, 2)
    action_size = game_data.start_states.shape[
        -2] * game_data.start_states.shape[-1] + 1
    qcomplete = _compute_qcomplete(partial_transition_value_logits,
                                   value_logits, sampled_actions, action_size)
    policy_loss, policy_acc, policy_entropy = _compute_policy_metrics(
        policy_logits, qcomplete)

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
        game_data: data.GameData,
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
        game_data: data.GameData,
        rng_key: jax.random.KeyArray) -> Tuple[optax.Params, LossMetrics]:
    """Computes the gradients of the loss function."""
    loss_fn = jax.value_and_grad(_extract_total_loss, argnums=1, has_aux=True)
    (_, loss_metrics), grads = loss_fn(go_model, params, game_data, rng_key)
    return grads, loss_metrics
