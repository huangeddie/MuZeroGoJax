"""Loss functions."""

import dataclasses
from typing import Tuple

import gojax
import haiku as hk
import jax.nn
import jax.tree_util
import optax
from absl import flags
from jax import lax
from jax import numpy as jnp

from muzero_gojax import data, game, models, nt_utils

_TEMPERATURE = flags.DEFINE_float(
    "temperature", 0.1,
    "Temperature for value labels in policy cross entropy loss.")
_HYPO_STEPS = flags.DEFINE_integer(
    'hypo_steps', 1,
    'Number of hypothetical steps to take for computing the losses.')
_SAMPLE_ACTION_SIZE = flags.DEFINE_integer(
    'sample_action_size', 2,
    'Number of actions to sample from for policy improvement.')
_ADD_DECODE_LOSS = flags.DEFINE_bool(
    "add_decode_loss", True,
    "Whether or not to add the decode loss to the total loss.")
_ADD_VALUE_LOSS = flags.DEFINE_bool(
    "add_value_loss", True,
    "Whether or not to add the value loss to the total loss.")
_ADD_POLICY_LOSS = flags.DEFINE_bool(
    "add_policy_loss", True,
    "Whether or not to add the policy loss to the total loss.")
_TRANS_LOSS = flags.DEFINE_enum("trans_loss", 'mse', ['mse', 'kl_div', 'bce'],
                                "Transition loss")
_ADD_TRANS_LOSS = flags.DEFINE_bool(
    "add_trans_loss", True,
    "Whether or not to add the transition loss to the total loss.")
_TRANSITION_FLOW = flags.DEFINE_bool(
    "transition_flow", False,
    "Let the gradient flow through the transition model.")


def _compute_decode_metrics(go_model: hk.MultiTransformed,
                            params: optax.Params, loss_data: data.LossData,
                            nt_mask: jnp.ndarray) -> data.SummedMetrics:
    """Updates the cumulative decode loss."""
    decode_model = go_model.apply[models.DECODE_INDEX]
    batch_size, traj_len = loss_data.nt_curr_embeds.shape[:2]
    flat_embeds = nt_utils.flatten_first_two_dims(loss_data.nt_curr_embeds)
    flat_decoded_states_logits = decode_model(params, None, flat_embeds)
    decoded_states_logits = nt_utils.unflatten_first_dim(
        flat_decoded_states_logits, batch_size, traj_len)
    decode_loss = nt_utils.nt_sigmoid_cross_entropy(
        decoded_states_logits,
        loss_data.trajectories.nt_states.astype(decoded_states_logits.dtype),
        nt_mask)
    decode_acc = jnp.nan_to_num(
        nt_utils.nt_sign_acc(decoded_states_logits,
                             loss_data.trajectories.nt_states * 2 - 1,
                             nt_mask))
    return data.SummedMetrics(loss=decode_loss,
                              acc=decode_acc,
                              steps=jnp.ones((), dtype='uint8'))


def _compute_value_metrics(go_model: hk.MultiTransformed, params: optax.Params,
                           loss_data: data.LossData,
                           nt_mask: jnp.ndarray) -> data.SummedMetrics:
    """Updates the cumulative value loss with rotation and flipping augmentation."""
    value_model = go_model.apply[models.VALUE_INDEX]
    batch_size, total_steps = loss_data.nt_curr_embeds.shape[:2]
    # N x C x H x W
    states = nt_utils.flatten_first_two_dims(loss_data.nt_curr_embeds)
    flat_value_logits = value_model(params, None, states)
    value_logits = jnp.reshape(flat_value_logits, (batch_size, total_steps))
    labels = (loss_data.nt_player_labels + 1) / jnp.array(
        2, dtype=flat_value_logits.dtype)
    val_loss = nt_utils.nt_sigmoid_cross_entropy(value_logits,
                                                 labels=labels,
                                                 nt_mask=nt_mask)
    val_acc = jnp.nan_to_num(
        nt_utils.nt_sign_acc(value_logits, loss_data.nt_player_labels,
                             nt_mask))
    return data.SummedMetrics(loss=val_loss,
                              acc=val_acc,
                              steps=jnp.ones((), dtype='uint8'))


def _update_curr_embeds(loss_data: data.LossData) -> data.SummedMetrics:
    """
    Updates the current embeddings.

    Stop the gradient for the transition embeddings.
    We don't want the transition model to change for the policy or value losses.
    """
    nt_hypo_embed_logits = _get_next_hypo_embed_logits(loss_data)
    if not _TRANSITION_FLOW.value:
        nt_hypo_embed_logits = lax.stop_gradient(nt_hypo_embed_logits)
    return loss_data.replace(nt_curr_embeds=nt_hypo_embed_logits)


def _compute_policy_metrics(go_model: hk.MultiTransformed,
                            params: optax.Params, loss_data: data.LossData,
                            nt_suffix_mask: jnp.ndarray) -> data.SummedMetrics:
    """Updates the policy loss."""
    nt_transitions = loss_data.nt_transition_logits
    batch_size, total_steps, action_size = nt_transitions.shape[:3]
    policy_model = go_model.apply[models.POLICY_INDEX]
    policy_logits = nt_utils.unflatten_first_dim(
        policy_model(params, None,
                     nt_utils.flatten_first_two_dims(
                         loss_data.nt_curr_embeds)), batch_size, total_steps)
    value_model = go_model.apply[models.VALUE_INDEX]
    transition_value_logits = nt_utils.unflatten_first_dim(
        value_model(params, None,
                    nt_utils.flatten_first_n_dim(nt_transitions, n_dim=3)),
        batch_size, total_steps, action_size)
    labels = lax.stop_gradient(policy_logits -
                               transition_value_logits / _TEMPERATURE.value)
    policy_loss = nt_utils.nt_categorical_cross_entropy(policy_logits,
                                                        labels,
                                                        nt_mask=nt_suffix_mask)
    policy_acc = nt_utils.nt_mask_mean(
        jnp.equal(jnp.argmax(policy_logits, axis=2), jnp.argmax(labels,
                                                                axis=2)),
        nt_suffix_mask).astype(policy_loss.dtype)
    policy_entropy = nt_utils.nt_entropy(policy_logits)
    return data.SummedMetrics(loss=policy_loss,
                              acc=policy_acc,
                              entropy=policy_entropy,
                              steps=jnp.ones((), dtype='uint8'))


def _update_trans_loss_and_metrics(loss_data: data.LossData,
                                   nt_mask: jnp.ndarray) -> data.LossData:
    """Updates the transition loss and accuracies."""

    nt_hypo_embed_logits = _get_next_hypo_embed_logits(loss_data)
    loss_fn = {
        'mse': nt_utils.nt_mse_loss,
        'kl_div': nt_utils.nt_kl_div_loss,
        'bce': nt_utils.nt_bce_loss
    }[_TRANS_LOSS.value]
    trans_loss = jnp.nan_to_num(
        loss_fn(nt_hypo_embed_logits, loss_data.nt_original_embeds, nt_mask))
    trans_acc = jnp.nan_to_num(
        nt_utils.nt_sign_acc(nt_hypo_embed_logits,
                             loss_data.nt_original_embeds, nt_mask))
    return data.SummedMetrics(loss=trans_loss,
                              acc=trans_acc,
                              steps=jnp.ones((), dtype='uint8'))


def _get_next_hypo_embed_logits(loss_data: data.LossData) -> jnp.ndarray:
    """
    Gets the next hypothetical logits from the transitions.

    :param loss_data: data.LossData.
    :return: An N x T x (D*) array.
    """
    batch_size, total_steps = loss_data.nt_transition_logits.shape[:2]
    flat_transitions = nt_utils.flatten_first_two_dims(
        loss_data.nt_transition_logits)
    flat_actions = nt_utils.flatten_first_two_dims(
        loss_data.trajectories.nt_actions)
    # taken_transitions: (N * T) x (D*)
    taken_transitions = flat_transitions[jnp.arange(len(flat_actions)),
                                         flat_actions]
    nt_hypo_embed_logits = jnp.roll(nt_utils.unflatten_first_dim(
        taken_transitions, batch_size, total_steps),
                                    shift=1,
                                    axis=1)
    return nt_hypo_embed_logits


def _update_transitions(go_model: hk.MultiTransformed, params: optax.Params,
                        loss_data: data.LossData) -> data.LossData:
    """Updates the N x T x A x C x H x W transition array."""
    batch_size, total_steps = loss_data.nt_curr_embeds.shape[:2]
    transition_model = go_model.apply[models.TRANSITION_INDEX]
    flattened_embeds = nt_utils.flatten_first_two_dims(
        loss_data.nt_curr_embeds)
    if not _TRANSITION_FLOW.value:
        flattened_embeds = lax.stop_gradient(flattened_embeds)
    # flat_transitions: (N * T) x A x D x H x W
    flat_transitions = transition_model(params, None, flattened_embeds)
    loss_data = loss_data.replace(
        nt_transition_logits=nt_utils.unflatten_first_dim(
            flat_transitions, batch_size, total_steps))
    return loss_data


def _update_k_step_losses(go_model: hk.MultiTransformed, params: optax.Params,
                          step_index: int,
                          loss_data: data.LossData) -> data.LossData:
    """
    Updates data to the i'th hypothetical step and adds the corresponding value and policy losses
    at that step.

    :param go_model: Haiku model architecture.
    :param params: Parameters of the model.
    :param step_index: The index of the hypothetical step (0-indexed).
    :param loss_data: The loss data. See `_initialize_loss_data`.
    :return: An updated version of the loss data.
    """
    batch_size, total_steps = loss_data.nt_curr_embeds.shape[:2]
    nt_suffix_mask = nt_utils.make_suffix_nt_mask(batch_size, total_steps,
                                                  total_steps - step_index)

    train_metrics: data.TrainMetrics = loss_data.train_metrics
    train_metrics = train_metrics.update_decode(
        _compute_decode_metrics(go_model, params, loss_data, nt_suffix_mask))
    train_metrics = train_metrics.update_value(
        _compute_value_metrics(go_model, params, loss_data, nt_suffix_mask))
    # loss_data = _update_policy_logits(go_model, params, loss_data)
    # loss_data = _update_sampled_actions(go_model)
    loss_data = _update_transitions(go_model, params, loss_data)
    train_metrics = train_metrics.update_policy(
        _compute_policy_metrics(go_model, params, loss_data, nt_suffix_mask))

    # The transition loss / accuracy requires knowledge of the next transition, which is why our
    # suffix mask is one less than the other suffix masks.
    nt_suffix_minus_one_mask = nt_utils.make_suffix_nt_mask(
        batch_size, total_steps, total_steps - step_index - 1)
    train_metrics = train_metrics.update_trans(
        _update_trans_loss_and_metrics(loss_data, nt_suffix_minus_one_mask))

    loss_data = _update_curr_embeds(loss_data)
    # Since we updated the embeddings, the number of valid embeddings is one less than before.
    train_metrics = train_metrics.update_decode(
        _compute_decode_metrics(go_model, params, loss_data, nt_suffix_mask))
    train_metrics = train_metrics.update_value(
        _compute_value_metrics(go_model, params, loss_data, nt_suffix_mask))
    return loss_data.replace(train_metrics=train_metrics)


def _initialize_loss_data(trajectories: data.Trajectories,
                          embeddings: jnp.ndarray) -> data.LossData:
    """
    Returns a tracking dictionary of the loss data.
    :param trajectories: A dictionary of states and actions.
    :param embeddings: Embeddings of the states in the trajectories.
    :return: a data.LossData structure.
    """
    nt_states = trajectories.nt_states
    trajectories = data.Trajectories(nt_states=nt_states,
                                     nt_actions=trajectories.nt_actions)
    batch_size, total_steps = nt_states.shape[:2]
    board_size = nt_states.shape[-1]
    embed_dim = embeddings.shape[2]
    action_size = gojax.get_action_size(nt_states)
    nt_transition_logits = jnp.zeros((batch_size, total_steps, action_size,
                                      embed_dim, board_size, board_size),
                                     dtype=embeddings.dtype)
    sample_action_size = action_size
    if _SAMPLE_ACTION_SIZE.value is not None and _SAMPLE_ACTION_SIZE.value > 0:
        sample_action_size = _SAMPLE_ACTION_SIZE.value
        if sample_action_size > action_size:
            raise ValueError(
                f'Sample action size {_SAMPLE_ACTION_SIZE.value} '
                f'is greater than full action size {action_size}.')

    train_metrics = data.init_train_metrics(embeddings.dtype)
    nt_player_labels = game.get_nt_player_labels(nt_states)
    train_metrics = dataclasses.replace(train_metrics,
                                        win_rates=game.get_win_rates(
                                            nt_player_labels,
                                            embeddings.dtype))
    nt_sampled_actions = jnp.full(
        (batch_size, total_steps, sample_action_size),
        fill_value=-1,
        dtype='uint16')
    return data.LossData(trajectories=trajectories,
                         nt_original_embeds=embeddings,
                         nt_curr_embeds=embeddings,
                         nt_sampled_actions=nt_sampled_actions,
                         nt_transition_logits=nt_transition_logits,
                         nt_player_labels=nt_player_labels,
                         train_metrics=train_metrics)


def _compute_k_step_losses(go_model: hk.MultiTransformed, params: optax.Params,
                           trajectories: data.Trajectories) -> data.LossData:
    """
    Computes the value, and policy k-step losses.

    :param go_model: Haiku model architecture.
    :param params: Parameters of the model.
    :param trajectories: An N x T X C X H x W boolean array.
    :return: A dictionary of cumulative losses and model state
    """
    embed_model = go_model.apply[models.EMBED_INDEX]
    nt_states = trajectories.nt_states
    batch_size, total_steps = nt_states.shape[:2]
    embeddings = nt_utils.unflatten_first_dim(
        embed_model(params, None, nt_utils.flatten_first_two_dims(nt_states)),
        batch_size, total_steps)
    compute_step_fn = jax.tree_util.Partial(_update_k_step_losses, go_model,
                                            params)
    init_data = _initialize_loss_data(trajectories, embeddings)
    if _HYPO_STEPS.value > 1:
        return lax.fori_loop(lower=0,
                             upper=_HYPO_STEPS.value,
                             body_fun=compute_step_fn,
                             init_val=init_data)
    else:
        return compute_step_fn(0, init_data)


def _aggregate_k_step_losses(
        go_model: hk.MultiTransformed, params: optax.Params,
        trajectories: data.Trajectories
) -> Tuple[jnp.ndarray, data.TrainMetrics]:
    """
    Computes the sum of all losses.

    Use this function to compute the gradient of the model parameters.

    :param go_model: Haiku model architecture.
    :param params: Parameters of the model.
    :param trajectories: An N x T X C X H x W boolean array.
    :return: The total loss, and metrics.
    """
    loss_data: data.LossData = _compute_k_step_losses(go_model, params,
                                                      trajectories)
    train_metrics = loss_data.train_metrics.average()
    total_loss = jnp.zeros((), dtype=train_metrics.value.loss.dtype)
    if _ADD_DECODE_LOSS.value:
        total_loss += train_metrics.decode.loss
    if _ADD_VALUE_LOSS.value:
        total_loss += train_metrics.value.loss
    if _ADD_POLICY_LOSS.value:
        total_loss += train_metrics.policy.loss
    if _ADD_TRANS_LOSS.value:
        total_loss += train_metrics.trans.loss
    return total_loss, train_metrics


def compute_loss_gradients_and_metrics(
        go_model: hk.MultiTransformed, params: optax.Params,
        trajectories: data.Trajectories
) -> Tuple[optax.Params, data.TrainMetrics]:
    """Computes the gradients of the loss function."""
    loss_fn = jax.value_and_grad(_aggregate_k_step_losses,
                                 argnums=1,
                                 has_aux=True)
    (_, metric_data), grads = loss_fn(go_model, params, trajectories)
    return grads, metric_data
