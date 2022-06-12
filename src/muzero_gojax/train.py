"""Manages the MuZero training of Go models."""
import logging

import gojax
import jax.nn
import jax.numpy as jnp
import jax.random
from jax import jit
from jax import lax

from muzero_gojax import models
from muzero_gojax.game import get_actions_and_labels
from muzero_gojax.game import self_play


def nd_categorical_cross_entropy(x_logits, y_logits, temp=None, mask=None):
    """
    Categorical cross-entropy with respect to the last dimension.

    :param x_logits: (N*) x D float array
    :param y_logits: (N*) x D float array
    :param temp: temperature constant
    :param mask: 0-1 mask to determine which logits to consider.
    :return: Mean cross-entropy loss between the softmax of x and softmax of (y / temp)
    """
    if temp is None:
        temp = 1
    if mask is None:
        mask = jnp.ones(x_logits.shape[:-1])
    cross_entropy = -jnp.sum(
        jax.nn.softmax(y_logits / temp) * jax.nn.log_softmax(x_logits),
        axis=-1)

    return jnp.sum(cross_entropy * mask) / jnp.sum(mask, dtype=float)


def sigmoid_cross_entropy(value_logits, labels, mask=None):
    """
    Computes the sigmoid cross-entropy given binary labels and logit values.

    :param value_logits: N^D array of float values
    :param labels: N^D array of N binary (0, 1) values
    :param mask: 0-1 mask to determine which logits to consider.
    :return: Mean cross-entropy loss between the sigmoid of the value logits and the labels.
    """
    if mask is None:
        mask = jnp.ones_like(value_logits)
    cross_entropy = -labels * jax.nn.log_sigmoid(value_logits) - (1 - labels) * jax.nn.log_sigmoid(
        -value_logits)
    return jnp.sum(cross_entropy * mask) / jnp.sum(mask, dtype=float)


def make_first_k_steps_mask(batch_size, total_steps, k):
    """Creates a boolean mask of shape batch_size x total_steps, where the first k steps are True
    and the rest are false."""
    return jnp.repeat(jnp.expand_dims(jnp.arange(total_steps) < k, 0), batch_size, axis=0)


def _compute_policy_loss(policy_model, value_model, params, i, transitions, nt_embeddings):
    # pylint: disable=too-many-arguments
    batch_size, total_steps, action_size = transitions.shape[:3]
    embed_shape = transitions.shape[3:]
    num_examples = batch_size * total_steps
    # transition_value_logits is a 1-D vector of length N * T * A.
    flat_transition_value_logits = value_model(params, None,
                                               jnp.reshape(transitions, (
                                                   num_examples * action_size,) + embed_shape))
    trajectory_policy_shape = (batch_size, total_steps, action_size)
    transition_value_logits = jnp.reshape(flat_transition_value_logits,
                                          trajectory_policy_shape)
    policy_logits = policy_model(params, None,
                                 jnp.reshape(nt_embeddings, (num_examples,) + embed_shape))
    return nd_categorical_cross_entropy(
        jnp.reshape(policy_logits, trajectory_policy_shape),
        transition_value_logits,
        mask=make_first_k_steps_mask(batch_size, total_steps, total_steps - i))


def _compute_value_loss(value_model, params, i, data):
    batch_size, total_steps = data['nt_embeds'].shape[:2]
    embed_shape = data['nt_embeds'].shape[2:]
    num_examples = batch_size * total_steps
    labels = (jnp.roll(data['nt_game_winners'], shift=i) + 1) / 2
    flat_value_logits = value_model(params, None,
                                    jnp.reshape(data['nt_embeds'], (num_examples,) + embed_shape))
    return sigmoid_cross_entropy(jnp.reshape(flat_value_logits, (batch_size, total_steps)), labels,
                                 mask=make_first_k_steps_mask(batch_size,
                                                              total_steps,
                                                              total_steps - i))


def update_k_step_losses(model_fn, params, i, data):
    """
    Updates data to the i'th hypothetical step and adds the corresponding value and policy losses
    at that step.

    :param model_fn: Haiku model architecture.
    :param params: Parameters of the model.
    :param i: The index of the hypothetical step (0-indexed).
    :param data: A dictionary structure of the format
        'nt_embeds': An N x T x (D*) array of Go state embeddings.
        'nt_actions': An N x T non-negative integer array.
        'nt_game_winners': An integer array of length N. 1 = black won, 0 = tie, -1 = white won.
        'cum_val_loss': Cumulative value loss.
    :return: An updated version of data.
    """
    _, value_model, policy_model, transition_model = model_fn.apply
    batch_size, total_steps = data['nt_embeds'].shape[:2]
    num_examples = batch_size * total_steps
    embed_shape = data['nt_embeds'].shape[2:]

    # Update the cumulative value loss.
    data['cum_val_loss'] += _compute_value_loss(value_model, params, i, data)

    # Get the transitions.
    # Flattened transitions is (N * T) x A x (D*)
    flat_transitions = transition_model(params, None, jnp.reshape(data['nt_embeds'], (
        num_examples,) + embed_shape))
    transitions = jnp.reshape(flat_transitions,
                              (batch_size, total_steps, flat_transitions.shape[1]) + embed_shape)

    # Update the cumulative policy loss.
    data['cum_policy_loss'] += _compute_policy_loss(policy_model, value_model, params, i,
                                                    transitions, data['nt_embeds'])

    # Update the state embeddings from the transitions indexed by the played actions.
    flat_next_states = flat_transitions[
        jnp.arange(num_examples), jnp.reshape(data['nt_actions'],
                                              num_examples)]
    data['nt_embeds'] = jnp.roll(
        jnp.reshape(flat_next_states, (batch_size, total_steps) + embed_shape), -1, axis=1)

    return data


def compute_k_step_losses(model_fn, params, trajectories, actions, game_winners, k=1):
    # pylint: disable=too-many-arguments
    """
    Computes the value, and policy k-step losses.

    :param model_fn: Haiku model architecture.
    :param params: Parameters of the model.
    :param trajectories: An N x T X C X H x W boolean array.
    :param actions: An N x T non-negative integer array.
    :param game_winners: An N x T integer array of length N. 1 = black won, 0 = tie, -1 = white won.
    :param k: Number of hypothetical steps.
    :return: A dictionary of cumulative losses.
    """
    embed_model = model_fn.apply[0]
    batch_size, total_steps, channels, height, width = trajectories.shape
    embeddings = embed_model(params, None, jnp.reshape(trajectories, (
        batch_size * total_steps, channels, height, width)))
    embed_shape = embeddings.shape[1:]
    data = lax.fori_loop(lower=0, upper=k,
                         body_fun=jax.tree_util.Partial(update_k_step_losses, model_fn, params),
                         init_val={'nt_embeds': jnp.reshape(embeddings, (
                             batch_size, total_steps) + embed_shape),
                                   'nt_actions': actions,
                                   'nt_game_winners': game_winners,
                                   'cum_val_loss': 0,
                                   'cum_policy_loss': 0})
    return {key: data[key] for key in ['cum_val_loss', 'cum_policy_loss']}


def compute_k_step_total_loss(model_fn, params, trajectories, actions, game_winners, k=1):
    # pylint: disable=too-many-arguments
    """
    Computes the sum of all losses.

    Use this function to compute the gradient of the model parameters.

    :param model_fn: Haiku model architecture.
    :param params: Parameters of the model.
    :param trajectories: An N x T X C X H x W boolean array.
    :param actions: An N x T non-negative integer array.
    :param game_winners: An N x T integer array of length N. 1 = black won, 0 = tie, -1 = white won.
    :param k: Number of hypothetical steps.
    :return: A dictionary of cumulative losses.
    """
    loss_dict = compute_k_step_losses(model_fn, params, trajectories, actions, game_winners, k)
    return loss_dict['cum_val_loss'] + loss_dict['cum_policy_loss']


def train_step(model_fn, params, trajectories, actions, game_winners, learning_rate):
    # pylint: disable=too-many-arguments
    """Updates the model in a single train_model step."""
    # K-step value loss and gradient.
    total_loss, grads = jax.value_and_grad(compute_k_step_total_loss, argnums=1)(model_fn, params,
                                                                                 trajectories,
                                                                                 actions,
                                                                                 game_winners)
    # Update parameters.
    params = jax.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    # Return updated parameters and loss metrics.
    return params, {'total_loss': total_loss}


def train_model(model_fn, batch_size, board_size, training_steps, max_num_steps, learning_rate,
                rng_key,
                use_jit):
    # pylint: disable=too-many-arguments
    """
    Trains the model with the specified hyperparameters.

    :param model_fn: JAX-Haiku model architecture.
    :param batch_size: Batch size.
    :param board_size: Board size.
    :param training_steps: Training steps.
    :param max_num_steps: Maximum number of steps.
    :param learning_rate: Learning rate.
    :param rng_key: RNG key.
    :param use_jit: Whether to use JIT compilation.
    :return: The model parameters.
    """
    self_play_fn = jit(self_play, static_argnums=(0, 2, 3, 4)) if use_jit else self_play
    get_actions_and_labels_fn = jit(get_actions_and_labels) if use_jit else get_actions_and_labels
    train_step_fn = jit(train_step, static_argnums=0) if use_jit else train_step

    logging.info("Initializing model...")
    params = model_fn.init(rng_key, gojax.new_states(board_size, 1))
    for _ in range(training_steps):
        logging.info('Self-playing...')
        trajectories = self_play_fn(model_fn, params, batch_size, board_size, max_num_steps,
                                    rng_key)
        logging.info('Self-play complete.')
        actions, game_winners = get_actions_and_labels_fn(trajectories)
        logging.info('Executing training step...')
        params, loss_metrics = train_step_fn(model_fn, params, trajectories, actions, game_winners,
                                             learning_rate)
        logging.info(f'Loss metrics: {loss_metrics}')

    return params


def train_from_flags(absl_flags):
    """Program entry point and highest-level algorithm flow of MuZero Go."""
    logging.info("Making model...")
    go_model = models.make_model(absl_flags.board_size, absl_flags.embed_model,
                                 absl_flags.value_model,
                                 absl_flags.policy_model,
                                 absl_flags.transition_model)
    logging.info("Training model...")
    rng_key = jax.random.PRNGKey(absl_flags.random_seed)
    _ = train_model(go_model, absl_flags.batch_size, absl_flags.board_size,
                    absl_flags.training_steps,
                    absl_flags.max_num_steps, absl_flags.learning_rate, rng_key, absl_flags.use_jit)
    logging.info("Training complete!")
    # TODO: Save the parameters in a specified flag directory defaulted to /tmp.
