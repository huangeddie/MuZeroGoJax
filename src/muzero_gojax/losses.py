"""Loss functions."""

from typing import Any
from typing import Callable
from typing import Tuple

import haiku as hk
import jax.nn
import jax.tree_util
import numpy as np
import optax
from absl import flags
from jax import lax
from jax import numpy as jnp

from muzero_gojax import game
from muzero_gojax import models


def _flatten_nt_dim(array: jnp.ndarray) -> jnp.ndarray:
    """Flatten the first two dimensions of the array."""
    assert jnp.ndim(array) >= 2
    return jnp.reshape(array, (np.prod(array.shape[:2]), *array.shape[2:]))


def nt_categorical_cross_entropy(x_logits: jnp.ndarray, y_logits: jnp.ndarray, temp: float = None,
                                 nt_mask: jnp.ndarray = None):
    """
    Categorical cross-entropy with respect to the last dimension.

    :param x_logits: N x T float array
    :param y_logits: N x T float array
    :param temp: temperature constant
    :param nt_mask: 0-1 mask to determine which logits to consider.
    :return: Mean cross-entropy loss between the softmax of x and softmax of (y / temp)
    """
    if temp is None:
        temp = 1
    if nt_mask is None:
        nt_mask = jnp.ones(x_logits.shape[:-1])
    cross_entropy = -jnp.sum(jax.nn.softmax(y_logits / temp) * jax.nn.log_softmax(x_logits),
                             axis=-1)

    return jnp.sum(cross_entropy * nt_mask) / jnp.sum(nt_mask, dtype='bfloat16')


def nt_sigmoid_cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray, nt_mask: jnp.ndarray = None):
    """
    Computes the sigmoid cross-entropy given binary labels and logit values.

    :param logits: N x T x (D*) float array
    :param labels: N x T x (D*) integer array of binary (0, 1) values
    :param nt_mask: 0-1 mask to determine which logits to consider.
    :return: Mean cross-entropy loss between the sigmoid of the value logits and the labels.
    """
    if nt_mask is None:
        nt_mask = jnp.ones_like(logits)
    cross_entropy = -labels * jax.nn.log_sigmoid(logits) - (1 - labels) * jax.nn.log_sigmoid(
        -logits)
    if jnp.ndim(logits) > 2:
        reduce_axes = tuple(range(2, jnp.ndim(logits)))
        nt_losses = jnp.sum(cross_entropy, axis=reduce_axes)
    else:
        nt_losses = cross_entropy
    return jnp.sum(nt_losses * nt_mask) / jnp.sum(nt_mask, dtype='bfloat16')


def make_prefix_nt_mask(batch_size: int, total_steps: int, step: int) -> jnp.ndarray:
    """
    Creates a boolean mask of shape batch_size x total_steps,
    where the first `step` columns (0-index, exclusive) are True and the rest are False.

    For example, make_prefix_nt_mask(2, 2, 1) = [[True, False], [True, False]].
    """
    return jnp.repeat(jnp.expand_dims(jnp.arange(total_steps) < step, 0), batch_size, axis=0)


def make_suffix_nt_mask(batch_size: int, total_steps: int, suffix_len: int) -> jnp.ndarray:
    """
    Creates a boolean mask of shape batch_size x total_steps,
    where the last `step` columns (0-index, exclusive) are True and the rest are false.

    For example, make_suffix_nt_mask(2, 2, 1) = [[False, True], [False, True]].
    """
    return jnp.repeat(jnp.expand_dims(jnp.arange(total_steps - 1, -1, step=-1) < suffix_len, 0),
                      batch_size, axis=0)


def _get_nt_value_logits(value_model, params: optax.Params, nt_embeds: jnp.ndarray) -> jnp.ndarray:
    """Gets the value logits for each state."""
    batch_size, total_steps = nt_embeds.shape[:2]
    flat_value_logits = value_model(params, None, _flatten_nt_dim(nt_embeds))
    value_logits = jnp.reshape(flat_value_logits, (batch_size, total_steps))
    return value_logits


def compute_value_loss(value_logits: jnp.ndarray, nt_game_winners: jnp.ndarray,
                       nt_mask: jnp.ndarray):
    """
    Computes the binary cross entropy loss between sigmoid(value_model(nt_embeds)) and
    nt_game_winners.

    :param value_logits: N x T float array of value logits.
    :param nt_game_winners: An N x T integer array of length N. 1 = black won, 0 = tie,
    -1 = white won.
    :param nt_mask: N x T boolean array to mask which losses we care about.
    :return: Scalar float value, and updated model state.
    """
    return nt_sigmoid_cross_entropy(value_logits,
                                    labels=(nt_game_winners + 1) / jnp.array(2, dtype='bfloat16'),
                                    nt_mask=nt_mask)


def compute_policy_loss_from_transition_values(policy_logits: jnp.ndarray,
                                               transition_value_logits: jnp.ndarray,
                                               nt_mask: jnp.ndarray, temp: float) -> jnp.ndarray:
    """
    Computes the softmax cross entropy loss using -value_model(transitions) as the labels and the
    policy_model(nt_embeddings) as the training logits.

    To prevent training the value model, the gradient flow is cut off from the value model.

    :param policy_logits: N x T x A policy logits.
    :param transition_value_logits: N x T x A transition value logits.
    :param nt_mask: N x T boolean array to mask which losses we care about.
    :param temp: Temperature adjustment for value model labels.
    :return: Scalar float value and updated model state.
    """
    return nt_categorical_cross_entropy(policy_logits, -lax.stop_gradient(transition_value_logits),
                                        temp, nt_mask=nt_mask)


def kl_div_trans_loss(transition_embeds: jnp.ndarray, target_embeds: jnp.ndarray,
                      nt_mask: jnp.ndarray):
    """
    Computes the KL-divergence between the output of the transition and embed models.

    Cuts off the gradient-flow from the target_embeds.
    We want the transition model to act like the embedding model.

    :param transition_embeds: N x T x (D*) float array.
    :param target_embeds: N x T x (D*) float array.
    :param nt_mask: N x T boolean array.
    :return: scalar float.
    """
    reduce_axes = tuple(range(2, len(transition_embeds.shape)))
    log_softmax_transition_embeds = jax.nn.log_softmax(transition_embeds.astype('bfloat16'),
                                                       axis=reduce_axes)
    softmax_target_embeds = lax.stop_gradient(
        jax.nn.softmax(target_embeds.astype('bfloat16'), axis=reduce_axes))
    log_softmax_target_embeds = lax.stop_gradient(
        jax.nn.log_softmax(target_embeds.astype('bfloat16'), axis=reduce_axes))
    nt_target_entropy = -jnp.sum(log_softmax_target_embeds * softmax_target_embeds,
                                 axis=reduce_axes)
    nt_losses = -jnp.sum(log_softmax_transition_embeds * softmax_target_embeds,
                         axis=reduce_axes) - lax.stop_gradient(nt_target_entropy)
    return jnp.sum(nt_losses * nt_mask) / jnp.sum(nt_mask, dtype='bfloat16')


def mse_trans_loss(transition_embeds: jnp.ndarray, target_embeds: jnp.ndarray,
                   nt_mask: jnp.ndarray):
    """
    Computes the mean-squared-error between the output of the transition and embed models.

    Cuts off the gradient-flow from the target_embeds.
    We want the transition model to act like the embedding model.

    :param transition_embeds: N x T x (D*) float array.
    :param target_embeds: N x T x (D*) float array.
    :param nt_mask: N x T boolean array.
    :return: scalar float.
    """
    reduce_axes = tuple(range(2, len(transition_embeds.shape)))
    nt_losses = 0.5 * jnp.sum(jnp.square(transition_embeds - lax.stop_gradient(target_embeds)),
                              axis=reduce_axes)
    return jnp.sum(nt_losses * nt_mask) / jnp.sum(nt_mask, dtype='bfloat16')


def bce_trans_loss(transition_embeds: jnp.ndarray, target_embeds: jnp.ndarray,
                   nt_mask: jnp.ndarray):
    """
    Computes the binary cross-entropy loss between the output of the transition and embed models.

    Cuts off the gradient-flow from the target_embeds.
    We want the transition model to act like the embedding model.

    :param transition_embeds: N x T x (D*) float array.
    :param target_embeds: N x T x (D*) float array.
    :param nt_mask: N x T boolean array.
    :return: scalar float.
    """
    reduce_axes = tuple(range(2, len(transition_embeds.shape)))
    log_p = jax.nn.log_sigmoid(transition_embeds)
    log_not_p = jax.nn.log_sigmoid(-transition_embeds)
    labels = lax.stop_gradient(target_embeds)
    nt_losses = jnp.sum(-labels * log_p - (1. - labels) * log_not_p, axis=reduce_axes)
    return jnp.sum(nt_losses * nt_mask) / jnp.sum(nt_mask, dtype='bfloat16')


def bce_trans_logits_acc(transition_embeds_logits: jnp.ndarray, target_embeds: jnp.ndarray,
                         nt_mask: jnp.ndarray):
    """
    Computes the binary accuracy between the output of the transition and embed models.

    Only applicable for the identity embedding.

    :param transition_embeds_logits: N x T x (D*) float array.
    :param target_embeds: N x T x (D*) float array.
    :param nt_mask: N x T boolean array.
    :return: scalar float.
    """
    reduce_axes = tuple(range(2, len(transition_embeds_logits.shape)))
    nt_predictions = transition_embeds_logits > 0
    nt_acc = jnp.mean(nt_predictions == target_embeds.astype(bool), axis=reduce_axes,
                      dtype='bfloat16')
    return jnp.sum(nt_acc * nt_mask) / jnp.sum(nt_mask, dtype='bfloat16')


def _compute_k_step_trans_loss(trans_loss: str, nt_hypothetical_embeds: jnp.ndarray,
                               nt_original_embeds: jnp.ndarray,
                               nt_mask: jnp.ndarray) -> jnp.ndarray:
    """

    :param trans_loss: Transition loss id.
    :param nt_hypothetical_embeds: N x T x (D*) array of hypothetical Go embeddings.
    :param nt_original_embeds: Original embeddings.
    :param nt_mask: N x T boolean array.
    :return: transition loss.
    """
    loss_fn = {'mse': mse_trans_loss, 'kl_div': kl_div_trans_loss, 'bce': bce_trans_loss}[
        trans_loss]
    return jnp.nan_to_num(loss_fn(nt_hypothetical_embeds, nt_original_embeds, nt_mask))


def get_flat_trans_logits(transition_model: Callable[..., Any], params: optax.Params,
                          nt_embeds: jnp.ndarray) -> jnp.ndarray:
    """
    Given N x T embeddings, returns all transition embeddings flattened,
    and cuts off the gradient flow to the embedding input.

    :param transition_model: Transition model.
    :param params: Model parameters.
    :param nt_embeds: N x T x (D*) array.
    :return: (N * T) x A x (D*) array.
    """
    return transition_model(params, None, lax.stop_gradient(_flatten_nt_dim(nt_embeds)))


def update_k_step_losses(absl_flags: flags.FlagValues, go_model: hk.MultiTransformed,
                         params: optax.Params, i: int, data: dict) -> dict:
    """
    Updates data to the i'th hypothetical step and adds the corresponding value and policy losses
    at that step.

    :param absl_flags: Abseil flags.
    :param go_model: Haiku model architecture.
    :param params: Parameters of the model.
    :param i: The index of the hypothetical step (0-indexed).
    :param data: A dictionary structure of the format
        'nt_states': N x T x C x B x B boolean array of the original Go states.
        'nt_curr_embeds': An N x T x (D*) array of Go state embeddings.
        'flattened_actions': An (N * T) non-negative integer array.
        'nt_game_winners': An N x T integer array of length N. 1 = black won, 0 = tie, -1 = white
        won.
        'cum_decode_loss': Cumulative decode loss.
        'cum_val_loss': Cumulative value loss.
        'cum_policy_loss': Cumulative policy loss.
        'cum_trans_loss': Cumulative embed loss.
    :return: An updated version of data.
    """

    # Compute basic info.
    batch_size, total_steps = data['nt_curr_embeds'].shape[:2]
    nt_suffix_mask = make_suffix_nt_mask(batch_size, total_steps, total_steps - i)

    # Update the cumulative decode loss.
    data = update_cum_decode_loss(go_model, params, data, nt_suffix_mask)

    # Update the cumulative value loss.
    data = update_cum_value_loss(go_model, params, data, nt_suffix_mask)

    # Get the transitions.
    # Flattened transitions is (N * T) x A x (D*)
    flat_transitions = get_flat_trans_logits(go_model.apply[models.TRANSITION_INDEX], params,
                                             data['nt_curr_embeds'])

    # Update the state embeddings from the transitions indexed by the played actions.
    nt_hypothetical_embeds = jnp.roll(
        jnp.reshape(flat_transitions[jnp.arange(len(flat_transitions)), data['flattened_actions']],
                    data['nt_curr_embeds'].shape), shift=1, axis=1)

    # Compute the transition model's embedding loss.
    # The transition loss / accuracy requires knowledge of the next transition, which is why our
    # suffix mask is one less than the other suffix mask.
    nt_minus_one_suffix_mask = make_suffix_nt_mask(batch_size, total_steps, total_steps - i - 1)
    if absl_flags.add_trans_loss:
        data['cum_trans_loss'] += _compute_k_step_trans_loss(absl_flags.trans_loss,
                                                             nt_hypothetical_embeds,
                                                             data['nt_original_embeds'],
                                                             nt_minus_one_suffix_mask)
    if absl_flags.monitor_trans_acc:
        data['cum_trans_acc'] += jnp.nan_to_num(
            bce_trans_logits_acc(nt_hypothetical_embeds, data['nt_original_embeds'],
                                 nt_minus_one_suffix_mask))

    # Update the cumulative policy loss
    if absl_flags.sigmoid_trans:
        flat_transitions = jax.nn.sigmoid(flat_transitions)
        nt_hypothetical_embeds = jax.nn.sigmoid(nt_hypothetical_embeds)
    data['cum_policy_loss'] += compute_policy_loss_from_transition_values(
        policy_logits=_get_policy_logits(go_model, params, data),
        transition_value_logits=_get_transition_value_logits(go_model, params, flat_transitions,
                                                             data), nt_mask=nt_suffix_mask,
        temp=absl_flags.temperature)

    # Update the embeddings. Stop the gradient for the transition embeddings.
    # We don't want the transition model to change for the policy or value losses.
    data['nt_curr_embeds'] = lax.stop_gradient(nt_hypothetical_embeds)

    return data


def _get_transition_value_logits(go_model: hk.MultiTransformed, params: optax.Params,
                                 flat_transitions: jnp.ndarray, data: dict) -> jnp.ndarray:
    """Handles reshaping logic to get the transition value logits."""
    batch_size, total_steps = data['nt_curr_embeds'].shape[:2]
    value_model = go_model.apply[models.VALUE_INDEX]
    transition_value_logits = jnp.reshape(
        value_model(params, None, _flatten_nt_dim(flat_transitions)), (batch_size, total_steps, -1))
    return transition_value_logits


def _get_policy_logits(go_model: hk.MultiTransformed, params: optax.Params,
                       data: dict) -> jnp.ndarray:
    """Handles reshaping logic to get the policy logits."""
    policy_model = go_model.apply[models.POLICY_INDEX]
    batch_size, total_steps = data['nt_curr_embeds'].shape[:2]
    policy_logits = policy_model(params, None, _flatten_nt_dim(data['nt_curr_embeds']))
    return jnp.reshape(policy_logits, (batch_size, total_steps, -1))


def update_cum_decode_loss(go_model: hk.MultiTransformed, params: optax.Params, data: dict,
                           nt_mask: jnp.ndarray) -> dict:
    """Updates the cumulative decode loss."""
    decode_model = go_model.apply[models.DECODE_INDEX]
    batch_size, traj_len = data['nt_curr_embeds'].shape[:2]
    flat_embeds = _flatten_nt_dim(data['nt_curr_embeds'])
    flat_decoded_states_logits = decode_model(params, None, flat_embeds)
    decoded_states_logits = jnp.reshape(flat_decoded_states_logits,
                                        (batch_size, traj_len, *data['nt_states'].shape[2:]))
    data['cum_decode_loss'] += nt_sigmoid_cross_entropy(decoded_states_logits,
                                                        data['nt_states'].astype('bfloat16'),
                                                        nt_mask)
    return data


def update_cum_value_loss(go_model: hk.MultiTransformed, params: optax.Params, data: dict,
                          nt_mask: jnp.ndarray) -> dict:
    """Updates the cumulative value loss."""
    value_model = go_model.apply[models.VALUE_INDEX]
    data['cum_val_loss'] += compute_value_loss(
        _get_nt_value_logits(value_model, params, data['nt_curr_embeds']), data['nt_game_winners'],
        nt_mask)
    return data


def compute_k_step_losses(absl_flags: flags.FlagValues, go_model: hk.MultiTransformed,
                          params: optax.Params, trajectories: dict) -> dict:
    """
    Computes the value, and policy k-step losses.

    :param absl_flags: Abseil flags.
    :param go_model: Haiku model architecture.
    :param params: Parameters of the model.
    :param trajectories: An N x T X C X H x W boolean array.
    :return: A dictionary of cumulative losses and model state
    """
    embed_model = go_model.apply[models.EMBED_INDEX]
    nt_states = trajectories['nt_states']
    batch_size, total_steps = nt_states.shape[:2]
    embeddings = embed_model(params, None, _flatten_nt_dim(nt_states))
    embeddings = jnp.reshape(embeddings, (batch_size, total_steps, *embeddings.shape[1:]))
    data = lax.fori_loop(lower=0, upper=absl_flags.hypo_steps,
                         body_fun=jax.tree_util.Partial(update_k_step_losses, absl_flags, go_model,
                                                        params), init_val={
            'nt_states': nt_states, 'nt_curr_embeds': embeddings, 'nt_original_embeds': embeddings,
            'flattened_actions': _flatten_nt_dim(trajectories['nt_actions']),
            'nt_game_winners': game.get_labels(nt_states), 'cum_decode_loss': 0, 'cum_val_loss': 0,
            'cum_policy_loss': 0, 'cum_trans_loss': 0, 'cum_trans_acc': 0,

        })
    return {key: data[key] for key in
            ['cum_decode_loss', 'cum_val_loss', 'cum_policy_loss', 'cum_trans_loss',
             'cum_trans_acc']}


def aggregate_k_step_losses(absl_flags: flags.FlagValues, go_model: hk.MultiTransformed,
                            params: optax.Params, trajectories: dict) -> Tuple[jnp.ndarray, dict]:
    """
    Computes the sum of all losses.

    Use this function to compute the gradient of the model parameters.

    :param absl_flags: Abseil flags.
    :param go_model: Haiku model architecture.
    :param params: Parameters of the model.
    :param trajectories: An N x T X C X H x W boolean array.
    :return: The total loss, and a dictionary of each cumulative loss + the updated model state
    """
    metrics_data = compute_k_step_losses(absl_flags, go_model, params, trajectories)
    total_loss = + metrics_data['cum_val_loss'] + metrics_data['cum_policy_loss']
    if absl_flags.add_decode_loss:
        total_loss += metrics_data['cum_decode_loss']
    if absl_flags.add_trans_loss:
        total_loss += metrics_data['cum_trans_loss']
    if not absl_flags.monitor_trans_loss:
        del metrics_data['cum_trans_loss']
    if absl_flags.monitor_trans_acc:
        metrics_data['trans_acc'] = metrics_data['cum_trans_acc'] / absl_flags.hypo_steps
    del metrics_data['cum_trans_acc']
    return total_loss, metrics_data
