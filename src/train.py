"""Manages the MuZero training of Go models."""
import gojax
import jax.nn
import jax.numpy as jnp
from jax import lax

from game import self_play
from game import get_actions_and_labels


def compute_policy_loss(action_logits, transition_value_logits, temp=None, mask=None):
    """
    Categorical cross-entropy of the model's policy function simulated at K lookahead steps.

    :param action_logits: (N*) x A float array
    :param transition_value_logits: (N*) x A float array representing state values for each next
    state
    :param temp: temperature constant
    :param mask: 0-1 mask to determine which logits to consider.
    :return: Mean cross-entropy loss between the softmax of the action logits and (transition value
    logits / temp)
    """
    if temp is None:
        temp = 1
    if mask is None:
        mask = jnp.ones(action_logits.shape[:-1])
    cross_entropy = -jnp.sum(
        jax.nn.softmax(transition_value_logits / temp) * jax.nn.log_softmax(action_logits),
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


def update_k_step_losses(model_fn, params, i, data):
    """
    Updates data to the i'th hypothetical step and adds the corresponding value and policy losses
    at that step.

    :param model_fn: Haiku model architecture.
    :param params: Parameters of the model.
    :param i: The index of the hypothetical step (0-indexed).
    :param data: A dictionary structure of the format
        'embeds': A batch array of N Go state embeddings.
        'actions': A batch array of action indices.
        'game_winners': An integer array of length N. 1 = black won, 0 = tie, -1 = white won.
        'cum_val_loss': Cumulative value loss.
    :return: An updated version of data.
    """
    _, value_model, policy_model, transition_model = model_fn.apply
    value_logits = value_model(params, None, data['embeds'])
    action_logits = policy_model(params, None, data['embeds'])
    labels = (jnp.roll(data['game_winners'], shift=i) + 1) / 2

    # Update the cumulative value loss.
    batch_size, total_steps = data['embeds'].shape[:2]
    num_examples = batch_size * total_steps
    embed_shape = data['embeds'].shape[2:]
    data['cum_val_loss'] += sigmoid_cross_entropy(value_logits, labels,
                                                  mask=make_first_k_steps_mask(batch_size,
                                                                               total_steps,
                                                                               total_steps - i))

    # Update the state embeddings.
    flattened_transitions = transition_model(params, None, jnp.reshape(data['embeds'], (
        num_examples,) + embed_shape))
    batch_size = len(data['embeds'])
    flattened_next_states = flattened_transitions[
        jnp.arange(batch_size), jnp.reshape(data['actions'],
                                            num_examples)]
    data['embeds'] = jnp.roll(
        jnp.reshape(flattened_next_states, (batch_size, total_steps) + embed_shape), -1, axis=1)

    # Update the cumulative policy loss.
    action_size = flattened_transitions.shape[1]
    transition_value_logits = value_model(params, None, jnp.reshape(flattened_transitions, (
        num_examples * action_size,) + embed_shape))
    transition_value_logits = jnp.reshape(transition_value_logits, (batch_size, action_size))
    data['cum_policy_loss'] += compute_policy_loss(action_logits, transition_value_logits,
                                                   mask=make_first_k_steps_mask(batch_size,
                                                                                total_steps,
                                                                                total_steps - i -
                                                                                1))

    return data


def compute_k_step_losses(model_fn, params, trajectories, actions, game_winners, k=1):
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
    data = lax.fori_loop(lower=0, upper=k,
                         body_fun=jax.tree_util.Partial(update_k_step_losses, model_fn, params),
                         init_val={'embeds': embed_model(params, None, trajectories),
                                   'actions': actions,
                                   'game_winners': game_winners,
                                   'cum_val_loss': 0,
                                   'cum_policy_loss': 0})
    return {key: data[key] for key in ['cum_val_loss', 'cum_policy_loss']}


def compute_k_step_total_loss(model_fn, params, trajectories, actions, game_winners, k=1):
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


def train_step(model_fn, params, states, actions, game_winners, learning_rate):
    """Updates the model in a single train step."""
    # K-step value loss and gradient.
    total_loss, grads = jax.value_and_grad(compute_k_step_total_loss, argnums=1)(model_fn, params,
                                                                                 states,
                                                                                 actions,
                                                                                 game_winners)
    # Update parameters.
    params = jax.tree_multimap(lambda p, g: p - learning_rate * g, params, grads)
    # Return updated parameters and loss metrics.
    return params, {'total_loss': total_loss}


def train(model_fn, batch_size, board_size, training_steps, max_num_steps, learning_rate, rng_key):
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
    :return: The model parameters.
    """
    params = model_fn.init(rng_key, gojax.new_states(board_size, 1))
    for _ in range(training_steps):
        trajectories = self_play(model_fn, params, batch_size, board_size, max_num_steps, rng_key)
        actions, game_winners = get_actions_and_labels(trajectories)
        params, loss_metrics = train_step(model_fn, params, trajectories, game_winners,
                                          learning_rate)
        print(loss_metrics)

    return params
