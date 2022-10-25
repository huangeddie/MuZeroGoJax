"""Loss functions."""

from typing import NamedTuple
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

_TEMPERATURE = flags.DEFINE_float("temperature", 0.1,
                                  "Temperature for value labels in policy cross entropy loss.")
_HYPO_STEPS = flags.DEFINE_integer('hypo_steps', 1,
                                   'Number of hypothetical steps to take for computing the losses.')
_ADD_DECODE_LOSS = flags.DEFINE_bool("add_decode_loss", True,
                                     "Whether or not to add the decode loss to the total loss.")
_ADD_VALUE_LOSS = flags.DEFINE_bool("add_value_loss", True,
                                    "Whether or not to add the value loss to the total loss.")
_ADD_POLICY_LOSS = flags.DEFINE_bool("add_policy_loss", True,
                                     "Whether or not to add the policy loss to the total loss.")
_TRANS_LOSS = flags.DEFINE_enum("trans_loss", 'mse', ['mse', 'kl_div', 'bce'], "Transition loss")
_ADD_TRANS_LOSS = flags.DEFINE_bool("add_trans_loss", True,
                                    "Whether or not to add the transition loss to the total loss.")
_SIGMOID_TRANS = flags.DEFINE_bool("sigmoid_trans", False,
                                   "Apply sigmoid to the transitions when we compute the policy "
                                   "loss and update the nt_curr_embeds in _update_k_step_losses.")
_TRANSITION_FLOW = flags.DEFINE_bool("transition_flow", False,
                                     "Let the gradient flow through the transition model.")


class LossData(NamedTuple):
    """Tracking data for computing the losses."""
    trajectories: game.Trajectories = None
    nt_curr_embeds: jnp.ndarray = None
    nt_original_embeds: jnp.ndarray = None
    nt_transition_logits: jnp.ndarray = None
    nt_game_winners: jnp.ndarray = None

    cum_decode_loss: jnp.ndarray = 0
    cum_decode_acc: jnp.ndarray = 0
    cum_val_loss: jnp.ndarray = 0
    cum_val_acc: jnp.ndarray = 0
    cum_policy_loss: jnp.ndarray = 0
    cum_policy_entropy: jnp.ndarray = 0
    cum_policy_acc: jnp.ndarray = 0
    cum_trans_loss: jnp.ndarray = 0
    cum_trans_acc: jnp.ndarray = 0


def _update_cum_decode_loss(go_model: hk.MultiTransformed, params: optax.Params, data: LossData,
                            nt_mask: jnp.ndarray) -> LossData:
    """Updates the cumulative decode loss."""
    decode_model = go_model.apply[models.DECODE_INDEX]
    batch_size, traj_len = data.nt_curr_embeds.shape[:2]
    flat_embeds = nt_utils.flatten_first_two_dims(data.nt_curr_embeds)
    flat_decoded_states_logits = decode_model(params, None, flat_embeds)
    decoded_states_logits = nt_utils.unflatten_first_dim(flat_decoded_states_logits, batch_size,
                                                         traj_len)
    data = data._replace(cum_decode_loss=data.cum_decode_loss + nt_utils.nt_sigmoid_cross_entropy(
        decoded_states_logits, data.trajectories.nt_states.astype('bfloat16'), nt_mask))
    data = data._replace(cum_decode_acc=data.cum_decode_acc + jnp.nan_to_num(
        nt_utils.nt_bce_logits_acc(decoded_states_logits, data.trajectories.nt_states, nt_mask)))
    return data


def _update_cum_value_loss(go_model: hk.MultiTransformed, params: optax.Params, data: LossData,
                           nt_mask: jnp.ndarray) -> LossData:
    """Updates the cumulative value loss with rotation and flipping augmentation."""
    value_model = go_model.apply[models.VALUE_INDEX]
    batch_size, total_steps = data.nt_curr_embeds.shape[:2]
    # N x C x H x W
    states = nt_utils.flatten_first_two_dims(data.nt_curr_embeds)
    flat_value_logits = value_model(params, None, states)
    value_logits = jnp.reshape(flat_value_logits, (batch_size, total_steps))
    labels = (data.nt_game_winners + 1) / jnp.array(2, dtype='bfloat16')
    data = data._replace(
        cum_val_loss=data.cum_val_loss + nt_utils.nt_sigmoid_cross_entropy(value_logits,
                                                                           labels=labels,
                                                                           nt_mask=nt_mask))
    data = data._replace(cum_val_acc=data.cum_val_acc + jnp.nan_to_num(
        nt_utils.nt_bce_logits_acc(value_logits, labels, nt_mask)))
    return data


def _update_curr_embeds(data: LossData) -> LossData:
    """
    Updates the current embeddings.

    Stop the gradient for the transition embeddings.
    We don't want the transition model to change for the policy or value losses.
    """
    nt_hypo_embed_logits = _get_next_hypo_embed_logits(data)
    if _SIGMOID_TRANS.value:
        nt_hypo_embeds = jax.nn.sigmoid(nt_hypo_embed_logits)
    else:
        nt_hypo_embeds = nt_hypo_embed_logits
    if not _TRANSITION_FLOW.value:
        nt_hypo_embeds = lax.stop_gradient(nt_hypo_embeds)
    data = data._replace(nt_curr_embeds=nt_hypo_embeds)
    return data


def _update_cum_policy_loss(go_model: hk.MultiTransformed, params: optax.Params, data: LossData,
                            nt_suffix_mask: jnp.ndarray) -> LossData:
    """Updates the policy loss."""
    if _SIGMOID_TRANS.value:
        nt_transitions = jax.nn.sigmoid(data.nt_transition_logits)
    else:
        nt_transitions = data.nt_transition_logits
    batch_size, total_steps, action_size = nt_transitions.shape[:3]
    policy_model = go_model.apply[models.POLICY_INDEX]
    policy_logits = nt_utils.unflatten_first_dim(
        policy_model(params, None, nt_utils.flatten_first_two_dims(data.nt_curr_embeds)),
        batch_size, total_steps)
    value_model = go_model.apply[models.VALUE_INDEX]
    transition_value_logits = nt_utils.unflatten_first_dim(
        value_model(params, None, nt_utils.flatten_first_n_dim(nt_transitions, n_dim=3)),
        batch_size, total_steps, action_size)
    labels = lax.stop_gradient(policy_logits - transition_value_logits / _TEMPERATURE.value)
    updated_cum_policy_loss = data.cum_policy_loss + nt_utils.nt_categorical_cross_entropy(
        policy_logits, labels, nt_mask=nt_suffix_mask)
    data = data._replace(cum_policy_loss=updated_cum_policy_loss)
    data = data._replace(
        cum_policy_entropy=data.cum_policy_entropy + nt_utils.nt_entropy(policy_logits))
    data = data._replace(cum_policy_acc=nt_utils.nt_mask_mean(
        jnp.equal(jnp.argmax(policy_logits, axis=2), jnp.argmax(labels, axis=2)), nt_suffix_mask))
    return data


def _update_trans_loss_and_metrics(data: LossData, nt_mask: jnp.ndarray) -> LossData:
    """Updates the transition loss and accuracies."""

    nt_hypo_embed_logits = _get_next_hypo_embed_logits(data)
    loss_fn = {
        'mse': nt_utils.nt_mse_loss, 'kl_div': nt_utils.nt_kl_div_loss, 'bce': nt_utils.nt_bce_loss
    }[_TRANS_LOSS.value]
    data = data._replace(cum_trans_loss=data.cum_trans_loss + jnp.nan_to_num(
        loss_fn(nt_hypo_embed_logits, data.nt_original_embeds, nt_mask)))

    # Update transition accuracy.
    binary_labels: jnp.ndarray
    if _SIGMOID_TRANS.value:
        binary_labels = data.nt_original_embeds > 0.5
    else:
        binary_labels = data.nt_original_embeds > 0
    data = data._replace(cum_trans_acc=data.cum_trans_acc + jnp.nan_to_num(
        nt_utils.nt_bce_logits_acc(nt_hypo_embed_logits, binary_labels, nt_mask)))
    return data


def _get_next_hypo_embed_logits(loss_data: LossData) -> jnp.ndarray:
    """
    Gets the next hypothetical logits from the transitions.

    :param loss_data: LossData.
    :return: An N x T x (D*) array.
    """
    batch_size, total_steps = loss_data.nt_transition_logits.shape[:2]
    flat_transitions = nt_utils.flatten_first_two_dims(loss_data.nt_transition_logits)
    flat_actions = nt_utils.flatten_first_two_dims(loss_data.trajectories.nt_actions)
    # taken_transitions: (N * T) x (D*)
    taken_transitions = flat_transitions[jnp.arange(len(flat_actions)), flat_actions]
    nt_hypo_embed_logits = jnp.roll(
        nt_utils.unflatten_first_dim(taken_transitions, batch_size, total_steps), shift=1, axis=1)
    return nt_hypo_embed_logits


def _update_transitions(go_model: hk.MultiTransformed, params: optax.Params,
                        data: LossData) -> LossData:
    """Updates the N x T x A x C x H x W transition array."""
    batch_size, total_steps = data.nt_curr_embeds.shape[:2]
    transition_model = go_model.apply[models.TRANSITION_INDEX]
    # flat_transitions: (N * T) x A x D x H x W
    flattened_embeds = nt_utils.flatten_first_two_dims(data.nt_curr_embeds)
    if not _TRANSITION_FLOW.value:
        flattened_embeds = lax.stop_gradient(flattened_embeds)
    flat_transitions = transition_model(params, None, flattened_embeds)
    data = data._replace(
        nt_transition_logits=nt_utils.unflatten_first_dim(flat_transitions, batch_size,
                                                          total_steps))
    return data


def _update_k_step_losses(go_model: hk.MultiTransformed, params: optax.Params, step_index: int,
                          data: LossData) -> LossData:
    """
    Updates data to the i'th hypothetical step and adds the corresponding value and policy losses
    at that step.

    :param go_model: Haiku model architecture.
    :param params: Parameters of the model.
    :param step_index: The index of the hypothetical step (0-indexed).
    :param data: The loss data. See `_initialize_loss_data`.
    :return: An updated version of the loss data.
    """
    batch_size, total_steps = data.nt_curr_embeds.shape[:2]
    nt_suffix_mask = nt_utils.make_suffix_nt_mask(batch_size, total_steps, total_steps - step_index)
    data = _update_cum_decode_loss(go_model, params, data, nt_suffix_mask)
    data = _update_cum_value_loss(go_model, params, data, nt_suffix_mask)
    data = _update_transitions(go_model, params, data)
    # The transition loss / accuracy requires knowledge of the next transition, which is why our
    # suffix mask is one less than the other suffix masks.
    data = _update_cum_policy_loss(go_model, params, data, nt_suffix_mask)

    nt_suffix_minus_one_mask = nt_utils.make_suffix_nt_mask(batch_size, total_steps,
                                                            total_steps - step_index - 1)
    data = _update_trans_loss_and_metrics(data, nt_suffix_minus_one_mask)

    data = _update_curr_embeds(data)
    # Since we updated the embeddings, the number of valid embeddings is one less than before.
    data = _update_cum_decode_loss(go_model, params, data, nt_suffix_minus_one_mask)
    data = _update_cum_value_loss(go_model, params, data, nt_suffix_minus_one_mask)
    return data


def _initialize_loss_data(trajectories: game.Trajectories, embeddings: jnp.ndarray) -> LossData:
    """
    Returns a tracking dictionary of the loss data.
    :param trajectories: A dictionary of states and actions.
    :param embeddings: Embeddings of the states in the trajectories.
    :return: a LossData structure.
    """
    nt_states = trajectories.nt_states
    trajectories = game.Trajectories(nt_states, trajectories.nt_actions)
    batch_size, total_steps = nt_states.shape[:2]
    board_size = nt_states.shape[-1]
    embed_dim = embeddings.shape[2]
    nt_transition_logits = jnp.zeros((
        batch_size, total_steps, gojax.get_action_size(nt_states), embed_dim, board_size,
        board_size), dtype='bfloat16')
    return LossData(trajectories, embeddings, embeddings, nt_transition_logits,
                    game.get_labels(nt_states))


def _compute_k_step_losses(go_model: hk.MultiTransformed, params: optax.Params,
                           trajectories: game.Trajectories) -> LossData:
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
        embed_model(params, None, nt_utils.flatten_first_two_dims(nt_states)), batch_size,
        total_steps)
    data: LossData = lax.fori_loop(lower=0, upper=_HYPO_STEPS.value,
                                   body_fun=jax.tree_util.Partial(_update_k_step_losses, go_model,
                                                                  params),
                                   init_val=_initialize_loss_data(trajectories, embeddings))
    return LossData(cum_decode_loss=data.cum_decode_loss, cum_decode_acc=data.cum_decode_acc,
                    cum_val_loss=data.cum_val_loss, cum_val_acc=data.cum_val_acc,
                    cum_policy_loss=data.cum_policy_loss, cum_policy_acc=data.cum_policy_acc,
                    cum_trans_loss=data.cum_trans_loss, cum_trans_acc=data.cum_trans_acc)


def _aggregate_k_step_losses(go_model: hk.MultiTransformed, params: optax.Params,
                             trajectories: game.Trajectories) -> Tuple[
    jnp.ndarray, metrics.Metrics]:
    """
    Computes the sum of all losses.

    Use this function to compute the gradient of the model parameters.

    :param go_model: Haiku model architecture.
    :param params: Parameters of the model.
    :param trajectories: An N x T X C X H x W boolean array.
    :return: The total loss, and metrics.
    """
    loss_data = _compute_k_step_losses(go_model, params, trajectories)
    total_loss = jnp.zeros((), dtype='bfloat16')
    metrics_data = metrics.Metrics()
    hypo_steps = _HYPO_STEPS.value
    if _ADD_DECODE_LOSS.value:
        total_loss += loss_data.cum_decode_loss
        metrics_data = metrics_data._replace(decode_acc=loss_data.cum_decode_acc / hypo_steps / 2)
        metrics_data = metrics_data._replace(decode_loss=loss_data.cum_decode_loss / hypo_steps / 2)
    if _ADD_VALUE_LOSS.value:
        total_loss += loss_data.cum_val_loss
        # We divide by two here because we update the cumulative value loss twice.
        # Once at the embedding, and another at the next embedding.
        metrics_data = metrics_data._replace(val_acc=loss_data.cum_val_acc / hypo_steps / 2)
        metrics_data = metrics_data._replace(val_loss=loss_data.cum_val_loss / hypo_steps / 2)
    if _ADD_POLICY_LOSS.value:
        total_loss += loss_data.cum_policy_loss
        metrics_data = metrics_data._replace(policy_acc=loss_data.cum_policy_acc / hypo_steps)
        metrics_data = metrics_data._replace(policy_loss=loss_data.cum_policy_loss / hypo_steps)
        metrics_data = metrics_data._replace(
            policy_entropy=loss_data.cum_policy_entropy / hypo_steps)
    if _ADD_TRANS_LOSS.value:
        total_loss += loss_data.cum_trans_loss
        metrics_data = metrics_data._replace(trans_acc=loss_data.cum_trans_acc / hypo_steps)
        metrics_data = metrics_data._replace(trans_loss=loss_data.cum_trans_loss / hypo_steps)
    return total_loss, metrics_data


def compute_loss_gradients_and_metrics(go_model: hk.MultiTransformed, params: optax.Params,
                                       trajectories: game.Trajectories) -> Tuple[
    optax.Params, metrics.Metrics]:
    """Computes the gradients of the loss function."""
    loss_fn = jax.value_and_grad(_aggregate_k_step_losses, argnums=1, has_aux=True)
    (_, metric_data), grads = loss_fn(go_model, params, trajectories)
    return grads, metric_data
