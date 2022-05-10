"""Manages the MuZero training of Go models."""
import gojax
import jax.nn
import jax.numpy as jnp

from game import self_play
from game import trajectories_to_dataset


def k_step_value_loss(model_fn, params, states, state_labels):
    """
    Sigmoid cross-entropy of the model's value function simulated at K lookahead steps.

    :param model_fn: Haiku model architecture.
    :param params: Parameters of the model.
    :param states: A batch array of N Go states.
    :param state_labels: An integer array of length N. 1 = black won, 0 = tie, -1 = white won.
    :return: A scalar loss.
    """
    _, value_logits, _ = model_fn.apply(params, rng=None, states=states)
    labels = (state_labels + 1) / 2
    return jnp.mean(-labels * jax.nn.log_sigmoid(value_logits) - (1 - labels) * jax.nn.log_sigmoid(
        -value_logits))


def k_step_policy_loss(model_fn, params, states, state_labels):
    raise NotImplementedError()


def train_step(model_fn, params, states, state_labels, learning_rate):
    """Updates the model in a single train step."""
    value_loss, value_grads = jax.value_and_grad(k_step_value_loss, argnums=1)(model_fn, params,
                                                                               states,
                                                                               state_labels)
    # TODO: Call the k_step_policy_loss
    params = jax.tree_multimap(lambda p, g: p - learning_rate * g, params, value_grads)
    return params, {'value_loss': value_loss}


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
        states, state_labels = trajectories_to_dataset(trajectories)
        params, _ = train_step(model_fn, params, states, state_labels, learning_rate)

    return params
