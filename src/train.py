"""Manages the MuZero training of Go models."""
import gojax
import jax.nn
import jax.numpy as jnp
from jax import lax

from game import self_play
from game import trajectories_to_dataset


def compute_policy_loss(action_logits, transition_value_logits, temp=None):
    """
    Categorical cross-entropy of the model's policy function simulated at K lookahead steps.

    :param action_logits: N x A float array
    :param transition_value_logits: N x A float array representing state values for each next state
    :param temp: temperature constant
    :return: Mean cross-entropy loss between the softmax of the action logits and (transition value
    logits / temp)
    """
    if temp is None:
        temp = 1
    return jnp.mean(
        -jnp.sum(jax.nn.softmax(transition_value_logits / temp) * jax.nn.log_softmax(action_logits),
                 axis=1))


def sigmoid_cross_entropy(value_logits, labels):
    """
    Computes the sigmoid cross-entropy given binary labels and logit values.

    :param value_logits: array of N float values
    :param labels: array of N binary (0, 1) values
    :return: Mean cross-entropy loss between the sigmoid of the value logits and the labels.
    """
    return jnp.mean(
        -labels * jax.nn.log_sigmoid(value_logits) - (1 - labels) * jax.nn.log_sigmoid(
            -value_logits))


def update_k_step_losses(model_fn, params, i, data):
    """
    Updates data to the i'th hypothetical step and adds the corresponding value and policy losses
    at that step.

    :param model_fn: Haiku model architecture.
    :param params: Parameters of the model.
    :param i: The index of the hypothetical step (0-indexed).
    :param data: A dictionary structure of the format
        'state_embeds': A batch array of N Go state embeddings.
        'actions': A batch array of action indices.
        'game_winners': An integer array of length N. 1 = black won, 0 = tie, -1 = white won.
        'cum_val_loss': Cumulative value loss.
    :return: An updated version of data.
    """
    _, value_model, policy_model, transition_model = model_fn.apply
    value_logits = value_model(params, None, data['state_embeds'])
    action_logits = policy_model(params, None, data['state_embeds'])
    labels = (jnp.roll(data['game_winners'], shift=i) + 1) / 2
    # TODO: Ignore the values that were rolled out.

    # Update the cumulative value loss
    data['cum_val_loss'] += sigmoid_cross_entropy(value_logits, labels)

    # Update the state embeddings
    transitions = transition_model(params, None, data['state_embeds'])
    batch_size = len(data['state_embeds'])
    data['state_embeds'] = jnp.roll(transitions[jnp.arange(batch_size), data['actions']], -1,
                                    axis=0)

    # Update the cumulative policy loss
    action_size, embed_shape = transitions.shape[1], transitions.shape[2:]
    transition_value_logits = value_model(params, None, jnp.reshape(transitions, (
        batch_size * action_size,) + embed_shape))
    transition_value_logits = jnp.reshape(transition_value_logits, (batch_size, action_size))
    data['cum_policy_loss'] += compute_policy_loss(action_logits, transition_value_logits)

    return data


def compute_k_step_losses(model_fn, params, states, actions, game_winners, k=1):
    """
    Computes the value, and policy k-step losses.

    :param model_fn: Haiku model architecture.
    :param params: Parameters of the model.
    :param states: A batch array of N Go states.
    :param actions: A batch array of action indices.
    :param game_winners: An integer array of length N. 1 = black won, 0 = tie, -1 = white won.
    :param k: Number of hypothetical steps.
    :return: A dictionary of cumulative losses.
    """
    embed_model = model_fn.apply[0]
    data = lax.fori_loop(lower=0, upper=k,
                         body_fun=jax.tree_util.Partial(update_k_step_losses, model_fn, params),
                         init_val={'state_embeds': embed_model(params, None, states),
                                   'actions': actions,
                                   'game_winners': game_winners,
                                   'cum_val_loss': 0,
                                   'cum_policy_loss': 0})
    return {key: data[key] for key in ['cum_val_loss', 'cum_policy_loss']}


def compute_k_step_total_loss(model_fn, params, states, actions, game_winners, k=1):
    """
    Computes the sum of all losses.

    Use this function to compute the gradient of the model parameters.

    :param model_fn: Haiku model architecture.
    :param params: Parameters of the model.
    :param states: A batch array of N Go states.
    :param actions: A batch array of action indices.
    :param game_winners: An integer array of length N. 1 = black won, 0 = tie, -1 = white won.
    :param k: Number of hypothetical steps.
    :return: A dictionary of cumulative losses.
    """
    loss_dict = compute_k_step_losses(model_fn, params, states, actions, game_winners, k)
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
        state_data, actions, game_winners = trajectories_to_dataset(trajectories)
        params, loss_metrics = train_step(model_fn, params, state_data, game_winners, learning_rate)
        print(loss_metrics)

    return params
