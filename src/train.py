"""Manages the MuZero training of Go models."""
import gojax

from game import self_play
from game import trajectories_to_dataset


def update_params(params, trajectories):
    """Updates the parameters in one gradient descent step for the given trajectories data."""
    _, _ = trajectories_to_dataset(trajectories)
    return params


def train(model_fn, batch_size, board_size, training_steps, max_num_steps, rng_key):
    # pylint: disable=too-many-arguments
    """
    Trains the model with the specified hyperparameters.

    :param model_fn: JAX-Haiku model architecture.
    :param batch_size: Batch size.
    :param board_size: Board size.
    :param training_steps: Training steps.
    :param max_num_steps: Maximum number of steps.
    :param rng_key: RNG key.
    :return: The model parameters.
    """
    params = model_fn.init(rng_key, gojax.new_states(board_size, 1))
    for _ in range(training_steps):
        trajectories = self_play(model_fn, params, batch_size, board_size, max_num_steps, rng_key)
        params = update_params(params, trajectories)

    return params
