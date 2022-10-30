"""Utility functions related to N x T x (D*) arrays."""
import jax.nn
import numpy as np
from jax import lax
from jax import numpy as jnp


def flatten_first_n_dim(array: jnp.ndarray, n_dim: int) -> jnp.ndarray:
    """Flattens the first n dimensions of the array."""
    return jnp.reshape(array, (np.prod(array.shape[:n_dim]), *array.shape[n_dim:]))


def flatten_first_two_dims(array: jnp.ndarray) -> jnp.ndarray:
    """Flatten the first two dimensions of the array."""
    return flatten_first_n_dim(array, n_dim=2)


def unflatten_first_dim(flattened_nt_array: jnp.array, *dims: int) -> jnp.ndarray:
    """Un-flattens the first dimension back into the batch size and total steps."""
    return jnp.reshape(flattened_nt_array, (*dims, *flattened_nt_array.shape[1:]))


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


def nt_mask_mean(nt_array: jnp.ndarray, nt_mask: jnp.ndarray) -> jnp.ndarray:
    """Averages only the elements with a corresponding non-zero mask."""
    return jnp.sum(nt_array * nt_mask) / jnp.sum(nt_mask, dtype='bfloat16')


def nt_categorical_cross_entropy(x_logits: jnp.ndarray, y_logits: jnp.ndarray,
                                 nt_mask: jnp.ndarray = None):
    """
    Categorical cross-entropy with respect to the last dimension.

    :param x_logits: N x T x A float array
    :param y_logits: N x T x A float array
    :param nt_mask: 0-1 mask to determine which logits to consider.
    :return: Mean cross-entropy loss between the softmax of x and softmax of (y / temp)
    """
    if nt_mask is None:
        nt_mask = jnp.ones(x_logits.shape[:-1])
    cross_entropy = -jnp.sum(jax.nn.softmax(y_logits)
                             * jax.nn.log_softmax(x_logits), axis=-1)

    return nt_mask_mean(cross_entropy, nt_mask)


def nt_entropy(logits: jnp.ndarray, nt_mask: jnp.ndarray = None):
    """
    Entropy with respect to the last dimension.

    :param logits: N x T x A float array
    :param nt_mask: 0-1 mask to determine which logits to consider.
    :return: Mean cross-entropy loss between the softmax of x and softmax of (y / temp)
    """
    if nt_mask is None:
        nt_mask = jnp.ones(logits.shape[:-1])
    entropy = -jnp.sum(jax.nn.softmax(logits) *
                       jax.nn.log_softmax(logits), axis=-1)

    return nt_mask_mean(entropy, nt_mask)


def nt_sigmoid_cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray, nt_mask: jnp.ndarray = None):
    """
    Computes the sigmoid cross-entropy given binary labels and logit values.

    :param logits: N x T x (D*) float array
    :param labels: N x T x (D*) integer array of binary (0, 1) values
    :param nt_mask: 0-1 mask to determine which logits to consider.
    :return: Mean cross-entropy loss between the sigmoid of the value logits and the labels.
    """
    if nt_mask is None:
        nt_mask = jnp.ones(logits.shape[:2], dtype=logits.dtype)
    cross_entropy = -labels * jax.nn.log_sigmoid(logits) - (1 - labels) * jax.nn.log_sigmoid(
        -logits)
    if jnp.ndim(logits) > 2:
        reduce_axes = tuple(range(2, jnp.ndim(logits)))
        nt_losses = jnp.mean(cross_entropy, axis=reduce_axes)
    else:
        nt_losses = cross_entropy
    return nt_mask_mean(nt_losses, nt_mask)


def nt_kl_div_loss(nt_logits: jnp.ndarray, target_embeds: jnp.ndarray, nt_mask: jnp.ndarray):
    """
    Computes the KL-divergence between the output of the transition and embed models.

    Cuts off the gradient-flow from the target_embeds.
    We want the transition model to act like the embedding model.

    :param nt_logits: N x T x (D*) float array.
    :param target_embeds: N x T x (D*) float array.
    :param nt_mask: N x T boolean array.
    :return: scalar float.
    """
    reduce_axes = tuple(range(2, len(nt_logits.shape)))
    log_softmax_transition_embeds = jax.nn.log_softmax(nt_logits.astype('bfloat16'),
                                                       axis=reduce_axes)
    softmax_target_embeds = lax.stop_gradient(
        jax.nn.softmax(target_embeds.astype('bfloat16'), axis=reduce_axes))
    log_softmax_target_embeds = lax.stop_gradient(
        jax.nn.log_softmax(target_embeds.astype('bfloat16'), axis=reduce_axes))
    nt_target_entropy = -jnp.sum(log_softmax_target_embeds * softmax_target_embeds,
                                 axis=reduce_axes)
    nt_losses = -jnp.sum(log_softmax_transition_embeds * softmax_target_embeds,
                         axis=reduce_axes) - lax.stop_gradient(nt_target_entropy)
    return nt_mask_mean(nt_losses, nt_mask)


def nt_mse_loss(nt_logits: jnp.ndarray, target_embeds: jnp.ndarray, nt_mask: jnp.ndarray):
    """
    Computes the mean-squared-error between the output of the transition and embed models.

    Cuts off the gradient-flow from the target_embeds.
    We want the transition model to act like the embedding model.

    :param nt_logits: N x T x (D*) float array.
    :param target_embeds: N x T x (D*) float array.
    :param nt_mask: N x T boolean array.
    :return: scalar float.
    """
    reduce_axes = tuple(range(2, len(nt_logits.shape)))
    nt_losses = 0.5 * jnp.mean(jnp.square(nt_logits - lax.stop_gradient(target_embeds)),
                               axis=reduce_axes)
    return nt_mask_mean(nt_losses, nt_mask)


def nt_bce_loss(nt_logits: jnp.ndarray, binary_target_embeds: jnp.ndarray, nt_mask: jnp.ndarray):
    """
    Computes the binary cross-entropy loss between the output of the transition and embed models.

    Cuts off the gradient-flow from the binary_target_embeds.
    We want the transition model to act like the embedding model.

    :param nt_logits: N x T x (D*) float array.
    :param binary_target_embeds: N x T x (D*) {0, 1} array.
    :param nt_mask: N x T boolean array.
    :return: scalar float.
    """
    reduce_axes = tuple(range(2, len(nt_logits.shape)))
    log_p = jax.nn.log_sigmoid(nt_logits)
    log_not_p = jax.nn.log_sigmoid(-nt_logits)
    labels = lax.stop_gradient(binary_target_embeds)
    nt_losses = jnp.sum(-labels * log_p - (1. - labels)
                        * log_not_p, axis=reduce_axes)
    return nt_mask_mean(nt_losses, nt_mask)


def nt_bce_logits_acc(nt_logits: jnp.ndarray, binary_target_embeds: jnp.ndarray,
                      nt_mask: jnp.ndarray):
    """
    Computes the binary accuracy between the output of the transition and embed models.

    Only applicable for the identity embedding.

    :param nt_logits: N x T x (D*) float array.
    :param binary_target_embeds: N x T x (D*) {0, 1} array.
    :param nt_mask: N x T boolean array.
    :return: scalar float.
    """
    reduce_axes = tuple(range(2, len(nt_logits.shape)))
    nt_predictions = nt_logits > 0
    nt_acc = jnp.mean(nt_predictions == binary_target_embeds.astype(bool), axis=reduce_axes,
                      dtype='bfloat16')
    return nt_mask_mean(nt_acc, nt_mask)
