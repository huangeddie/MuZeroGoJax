"""Loss functions."""

from collections import namedtuple
from typing import Tuple

import gojax
import haiku as hk
import jax.nn
import jax.tree_util
import optax
from absl import flags
from jax import lax
from jax import numpy as jnp

from muzero_gojax import game
from muzero_gojax import metrics
from muzero_gojax import models
from muzero_gojax import nt_utils

LossData = namedtuple('LossData', (
    'trajectories', 'nt_curr_embeds', 'nt_original_embeds', 'nt_transition_logits',
    'nt_game_winners', 'cum_decode_loss', 'cum_decode_acc', 'cum_val_loss', 'cum_val_acc',
    'cum_policy_loss', 'cum_trans_loss', 'cum_trans_acc'),
                      defaults=(game.Trajectories(), None, None, None, None, 0, 0, 0, 0, 0, 0, 0))


def update_cum_decode_loss(go_model: hk.MultiTransformed, params: optax.Params, data: LossData,
                           nt_mask: jnp.ndarray) -> LossData:
    """Updates the cumulative decode loss."""
    decode_model = go_model.apply[models.DECODE_INDEX]
    batch_size, traj_len = data.nt_curr_embeds.shape[:2]
    flat_embeds = nt_utils.flatten_nt_dim(data.nt_curr_embeds)
    flat_decoded_states_logits = decode_model(params, None, flat_embeds)
    decoded_states_logits = jnp.reshape(flat_decoded_states_logits, (
        batch_size, traj_len, *data.trajectories.nt_states.shape[2:]))
    data = data._replace(cum_decode_loss=data.cum_decode_loss + nt_utils.nt_sigmoid_cross_entropy(
        decoded_states_logits, data.trajectories.nt_states.astype('bfloat16'), nt_mask))
    data = data._replace(cum_decode_acc=data.cum_decode_acc + jnp.nan_to_num(
        nt_utils.nt_bce_logits_acc(decoded_states_logits, data.trajectories.nt_states, nt_mask)))
    return data


def update_cum_value_loss(go_model: hk.MultiTransformed, params: optax.Params, data: LossData,
                          nt_mask: jnp.ndarray) -> LossData:
    """Updates the cumulative value loss."""
    value_model = go_model.apply[models.VALUE_INDEX]
    batch_size, total_steps = data.nt_curr_embeds.shape[:2]
    flat_value_logits = value_model(params, None, nt_utils.flatten_nt_dim(data.nt_curr_embeds))
    value_logits = jnp.reshape(flat_value_logits, (batch_size, total_steps))
    labels = (data.nt_game_winners + 1) / jnp.array(2, dtype='bfloat16')
    data = data._replace(
        cum_val_loss=data.cum_val_loss + nt_utils.nt_sigmoid_cross_entropy(value_logits,
                                                                           labels=labels,
                                                                           nt_mask=nt_mask))
    data = data._replace(cum_val_acc=data.cum_val_acc + jnp.nan_to_num(
        nt_utils.nt_bce_logits_acc(value_logits,
                                   (data.nt_game_winners + 1) / jnp.array(2, dtype='bfloat16'),
                                   nt_mask)))
    return data


def update_curr_embeds(absl_flags: flags.FlagValues, data: LossData) -> LossData:
    """
    Updates the current embeddings.

    Stop the gradient for the transition embeddings.
    We don't want the transition model to change for the policy or value losses.
    """
    nt_hypo_embed_logits = get_next_hypo_embed_logits(data)
    if absl_flags.sigmoid_trans:
        nt_hypo_embeds = jax.nn.sigmoid(nt_hypo_embed_logits)
    else:
        nt_hypo_embeds = nt_hypo_embed_logits
    data = data._replace(nt_curr_embeds=lax.stop_gradient(nt_hypo_embeds))
    return data


def update_cum_policy_loss(absl_flags: flags.FlagValues, go_model: hk.MultiTransformed,
                           params: optax.Params, data: LossData,
                           nt_suffix_mask: jnp.ndarray) -> LossData:
    """Updates the policy loss."""
    if absl_flags.sigmoid_trans:
        nt_transitions = jax.nn.sigmoid(data.nt_transition_logits)
    else:
        nt_transitions = data.nt_transition_logits
    batch_size, total_steps, action_size = nt_transitions.shape[:3]
    policy_model = go_model.apply[models.POLICY_INDEX]
    policy_logits = nt_utils.unflatten_nt_dim(
        policy_model(params, None, nt_utils.flatten_nt_dim(data.nt_curr_embeds)), batch_size,
        total_steps)
    value_model = go_model.apply[models.VALUE_INDEX]
    transition_value_logits = nt_utils.unflatten_first_dim(
        value_model(params, None, nt_utils.flatten_first_n_dim(nt_transitions, n_dim=3)),
        batch_size, total_steps, action_size)
    updated_cum_policy_loss = data.cum_policy_loss + nt_utils.nt_categorical_cross_entropy(
        policy_logits, -lax.stop_gradient(transition_value_logits), absl_flags.temperature,
        nt_mask=nt_suffix_mask)
    data = data._replace(cum_policy_loss=updated_cum_policy_loss)
    return data


def _maybe_update_trans_loss_and_metrics(absl_flags: flags.FlagValues, data: LossData,
                                         curr_step: int) -> LossData:
    """Updates the transition loss and accuracies."""
    batch_size, total_steps = data.nt_curr_embeds.shape[:2]
    # The transition loss / accuracy requires knowledge of the next transition, which is why our
    # suffix mask is one less than the other suffix masks.
    nt_minus_one_suffix_mask = nt_utils.make_suffix_nt_mask(batch_size, total_steps,
                                                            total_steps - curr_step - 1)
    nt_hypo_embed_logits = get_next_hypo_embed_logits(data)
    if absl_flags.add_trans_loss:
        loss_fn = {
            'mse': nt_utils.nt_mse_loss, 'kl_div': nt_utils.nt_kl_div_loss,
            'bce': nt_utils.nt_bce_loss
        }[absl_flags.trans_loss]
        data = data._replace(cum_trans_loss=data.cum_trans_loss + jnp.nan_to_num(
            loss_fn(nt_hypo_embed_logits, data.nt_original_embeds, nt_minus_one_suffix_mask)))
    if absl_flags.monitor_trans_acc:
        data = data._replace(cum_trans_acc=data.cum_trans_acc + jnp.nan_to_num(
            nt_utils.nt_bce_logits_acc(nt_hypo_embed_logits, data.nt_original_embeds,
                                       nt_minus_one_suffix_mask)))
    return data


def get_next_hypo_embed_logits(data: LossData) -> jnp.ndarray:
    """
    Gets the next hypothetical logits from the transitions.

    :param data: A dictionary of model output data.
    :return: An N x T x (D*) array.
    """
    batch_size, total_steps = data.nt_transition_logits.shape[:2]
    flat_transitions = nt_utils.flatten_nt_dim(data.nt_transition_logits)
    flat_actions = nt_utils.flatten_nt_dim(data.trajectories.nt_actions)
    # taken_transitions: (N * T) x (D*)
    taken_transitions = flat_transitions[jnp.arange(len(flat_actions)), flat_actions]
    nt_hypo_embed_logits = jnp.roll(
        nt_utils.unflatten_nt_dim(taken_transitions, batch_size, total_steps), shift=1, axis=1)
    return nt_hypo_embed_logits


def _update_transitions(go_model: hk.MultiTransformed, params: optax.Params,
                        data: LossData) -> LossData:
    """Updates the N x T x A x C x H x W transition array."""
    batch_size, total_steps = data.nt_curr_embeds.shape[:2]
    transition_model = go_model.apply[models.TRANSITION_INDEX]
    # flat_transitions: (N * T) x A x D x H x W
    flat_transitions = transition_model(params, None, lax.stop_gradient(
        nt_utils.flatten_nt_dim(data.nt_curr_embeds)))
    data = data._replace(
        nt_transition_logits=nt_utils.unflatten_nt_dim(flat_transitions, batch_size, total_steps))
    return data


def update_k_step_losses(absl_flags: flags.FlagValues, go_model: hk.MultiTransformed,
                         params: optax.Params, i: int, data: LossData) -> LossData:
    """
    Updates data to the i'th hypothetical step and adds the corresponding value and policy losses
    at that step.

    :param absl_flags: Abseil flags.
    :param go_model: Haiku model architecture.
    :param params: Parameters of the model.
    :param i: The index of the hypothetical step (0-indexed).
    :param data: The loss data. See `_initialize_loss_data`.
    :return: An updated version of the loss data.
    """
    batch_size, total_steps = data.nt_curr_embeds.shape[:2]
    nt_suffix_mask = nt_utils.make_suffix_nt_mask(batch_size, total_steps, total_steps - i)
    data = update_cum_decode_loss(go_model, params, data, nt_suffix_mask)
    data = update_cum_value_loss(go_model, params, data, nt_suffix_mask)
    data = _update_transitions(go_model, params, data)
    data = _maybe_update_trans_loss_and_metrics(absl_flags, data, i)
    data = update_cum_policy_loss(absl_flags, go_model, params, data, nt_suffix_mask)
    data = update_curr_embeds(absl_flags, data)
    return data


def _initialize_loss_data(absl_flags: flags.FlagValues, trajectories: game.Trajectories,
                          embeddings: jnp.ndarray) -> LossData:
    """
    Returns a tracking dictionary of the loss data.
    :param absl_flags: Abseil flags.
    :param trajectories: A dictionary of states and actions.
    :param embeddings: Embeddings of the states in the trajectories.
    :return: a LossData structure.
    """
    nt_states = trajectories.nt_states
    trajectories = game.Trajectories(nt_states, trajectories.nt_actions)
    batch_size, total_steps = nt_states.shape[:2]
    board_size = nt_states.shape[-1]
    nt_transition_logits = jnp.zeros((
        batch_size, total_steps, gojax.get_action_size(nt_states), absl_flags.embed_dim, board_size,
        board_size), dtype='bfloat16')
    return LossData(trajectories, embeddings, embeddings, nt_transition_logits,
                    game.get_labels(nt_states))


def compute_k_step_losses(absl_flags: flags.FlagValues, go_model: hk.MultiTransformed,
                          params: optax.Params, trajectories: game.Trajectories) -> LossData:
    """
    Computes the value, and policy k-step losses.

    :param absl_flags: Abseil flags.
    :param go_model: Haiku model architecture.
    :param params: Parameters of the model.
    :param trajectories: An N x T X C X H x W boolean array.
    :return: A dictionary of cumulative losses and model state
    """
    embed_model = go_model.apply[models.EMBED_INDEX]
    nt_states = trajectories.nt_states
    batch_size, total_steps = nt_states.shape[:2]
    embeddings = nt_utils.unflatten_nt_dim(
        embed_model(params, None, nt_utils.flatten_nt_dim(nt_states)), batch_size, total_steps)
    data: LossData = lax.fori_loop(lower=0, upper=absl_flags.hypo_steps,
                                   body_fun=jax.tree_util.Partial(update_k_step_losses, absl_flags,
                                                                  go_model, params),
                                   init_val=_initialize_loss_data(absl_flags, trajectories,
                                                                  embeddings))
    return LossData(cum_decode_loss=data.cum_decode_loss, cum_decode_acc=data.cum_decode_acc,
                    cum_val_loss=data.cum_val_loss, cum_val_acc=data.cum_val_acc,
                    cum_policy_loss=data.cum_policy_loss, cum_trans_loss=data.cum_trans_loss,
                    cum_trans_acc=data.cum_trans_acc)


def aggregate_k_step_losses(absl_flags: flags.FlagValues, go_model: hk.MultiTransformed,
                            params: optax.Params, trajectories: game.Trajectories) -> Tuple[
    jnp.ndarray, metrics.Metrics]:
    """
    Computes the sum of all losses.

    Use this function to compute the gradient of the model parameters.

    :param absl_flags: Abseil flags.
    :param go_model: Haiku model architecture.
    :param params: Parameters of the model.
    :param trajectories: An N x T X C X H x W boolean array.
    :return: The total loss, and metrics.
    """
    loss_data = compute_k_step_losses(absl_flags, go_model, params, trajectories)
    total_loss = loss_data.cum_val_loss + loss_data.cum_policy_loss
    trans_acc = (
        loss_data.cum_trans_acc / absl_flags.hypo_steps if absl_flags.monitor_trans_acc else None)
    metrics_data = metrics.Metrics(decode_acc=loss_data.cum_decode_acc / absl_flags.hypo_steps,
                                   val_acc=loss_data.cum_val_acc / absl_flags.hypo_steps,
                                   trans_acc=trans_acc)
    if absl_flags.add_decode_loss:
        total_loss += loss_data.cum_decode_loss
    if absl_flags.add_trans_loss:
        total_loss += loss_data.cum_trans_loss
    return total_loss, metrics_data


def compute_loss_gradients_and_metrics(absl_flags: flags.FlagValues, go_model: hk.MultiTransformed,
                                       params: optax.Params, trajectories: game.Trajectories) -> \
        Tuple[optax.Params, metrics.Metrics]:
    """Computes the gradients of the loss function."""
    loss_fn = jax.value_and_grad(aggregate_k_step_losses, argnums=2, has_aux=True)
    (_, metric_data), grads = loss_fn(absl_flags, go_model, params, trajectories)
    return grads, metric_data
