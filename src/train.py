"""Manages the MuZero training of Go models."""
import gojax
import jax.nn
import jax.numpy as jnp
from jax import lax

from game import self_play
from game import trajectories_to_dataset


def value_loss(model_fn, params, i, data):
    """
    Sigmoid cross-entropy of the model's value function..

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
    _, value_model, _, transition_model = model_fn.apply
    value_logits = value_model(params, rng=None, state_embeds=data['state_embeds'])
    labels = (jnp.roll(data['game_winners'], shift=i) + 1) / 2
    # TODO: Ignore the values that were rolled out.
    data['cum_val_loss'] = jnp.mean(
        -labels * jax.nn.log_sigmoid(value_logits) - (1 - labels) * jax.nn.log_sigmoid(
            -value_logits))
    # Update the state embeddings
    transitions = transition_model(params, rng=None,
                                   state_embeds=data['state_embeds'])
    data['state_embeds'] = transitions[
        jnp.arange(len(data['state_embeds'])), jnp.roll(data['actions'], i)]

    return data


def k_step_value_loss(model_fn, params, states, actions, game_winners, k=1):
    """
    Sigmoid cross-entropy of the model's value function simulated at K lookahead steps.

    :param model_fn: Haiku model architecture.
    :param params: Parameters of the model.
    :param states: A batch array of N Go states.
    :param actions: A batch array of action indices.
    :param game_winners: An integer array of length N. 1 = black won, 0 = tie, -1 = white won.
    :param k: Number of hypothetical steps.
    :return: A scalar loss.
    """
    embed_model = model_fn.apply[0]
    data = lax.fori_loop(lower=0, upper=k,
                         body_fun=jax.tree_util.Partial(value_loss, model_fn, params),
                         init_val={'state_embeds': embed_model(params, rng=None, states=states),
                                   'actions': actions,
                                   'game_winners': game_winners,
                                   'cum_val_loss': 0})
    return data['cum_val_loss']


def k_step_policy_loss(model_fn, params, states, actions, game_winners):
    """Categorical cross-entropy of the model's policy function simulated at K lookahead steps."""
    # TODO: Implement and rename pylint.
    # pylint: disable=unused-argument
    raise NotImplementedError()


def train_step(model_fn, params, states, actions, game_winners, learning_rate):
    """Updates the model in a single train step."""
    # K-step value loss and gradient.
    value_loss, value_grads = jax.value_and_grad(k_step_value_loss, argnums=1)(model_fn, params,
                                                                               states,
                                                                               actions,
                                                                               game_winners)
    # TODO: K-step policy loss and gradient.
    policy_loss = 0
    # TODO: K-step transition loss and gradient.
    transition_loss = 0
    # Update parameters.
    params = jax.tree_multimap(lambda p, g: p - learning_rate * g, params, value_grads)
    # Return updated parameters and loss metrics.
    return params, {'value_loss': value_loss, 'policy_loss': policy_loss,
                    'transition_loss': transition_loss,
                    'total_loss': value_loss + policy_loss + transition_loss}


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
