"""Low-level informational functions about Go states."""

from typing import Union

from jax import numpy as jnp

from gojax import constants


def get_action_size(states):
    """
    The number of different actions to take.

    If states is N x B1 x B2, then the action size is B1 x B2 + 1.

    :param states: an array of N Go games.
    :return: a scalar integer.
    """
    b1, b2 = states.shape[-2:]
    return b1 * b2 + 1


def get_pieces_per_turn(states, turns):
    """
    Slices the black/white pieces of the states.

    See `at_pieces_per_turn` to get an update reference view of the pieces.

    :param states: an array of N Go games.
    :param turns: a boolean array of length N indicating which pieces to reference per state.
    :return: an array of shape N x B x B.
    """
    return states[jnp.arange(states.shape[0]), turns.astype('uint8')]


def get_turns(states):
    """
    Gets the turn for each state in states.

    :param states: a batch array of N Go games.
    :return: a boolean array of length N indicating whose turn it is for each state.
    """

    return jnp.alltrue(states[:, constants.TURN_CHANNEL_INDEX], axis=(1, 2))


def get_killed(states):
    """
    Gets the previously killed pieces for each state in states.

    :param states: a batch array of N Go games.
    :return: an N x B x B boolean array.
    """

    return states[:, constants.KILLED_CHANNEL_INDEX]


def get_passes(states):
    """
    Gets passes for each state in states.

    :param states: a batch array of N Go games.
    :return: a boolean array of length N indicating which state was passed.
    """

    return jnp.alltrue(states[:, constants.PASS_CHANNEL_INDEX], axis=(1, 2))


def get_ended(states):
    """
    Indicates which states have ended.

    :param states: a batch array of N Go games.
    :return: a boolean array of length N indicating which state ended.
    """
    return jnp.alltrue(states[:, constants.END_CHANNEL_INDEX], axis=(1, 2))


def get_empty_spaces(states, keepdims=False):
    """
    Gets the empty spaces for each state.

    :param states: a batch array of N Go games.
    :param keepdims: Whether to keep the Go state channel dimension.
    :return: an N x B x B boolean array or N x 1 x B x B boolean array.
    """
    return ~jnp.sum(
        states[:,
               [constants.BLACK_CHANNEL_INDEX, constants.WHITE_CHANNEL_INDEX]],
        axis=1,
        dtype=bool,
        keepdims=keepdims)


def get_occupied_spaces(states):
    """
    Gets the occupied spaces for each state (i.e. any black or white piecee).

    :param states: a batch array of N Go games.
    :return: an N x B x B boolean array.
    """
    return jnp.sum(
        states[:,
               [constants.BLACK_CHANNEL_INDEX, constants.WHITE_CHANNEL_INDEX]],
        axis=1,
        dtype=bool)


def at_pieces_per_turn(states, turns):
    """
    Update reference to the black/white pieces of the states.

    See `get_pieces_per_turn` to get a read-only view of the pieces.

    :param states: an array of N Go games.
    :param turns: a boolean array of length N indicating which pieces to reference per state.
    :return: an update reference array of shape N x B x B.
    """
    return states.at[jnp.arange(states.shape[0]), turns.astype('uint8')]


def at_location_per_turn(states, turns, row, col):
    """
    Update reference to specific turn-wise locations of the states.

    A more specific version of `at_pieces_per_turn`.

    :param states: an array of N Go games.
    :param turns: a boolean array of length N indicating which pieces to reference per state.
    :param row: integer row index.
    :param col: integer column index.
    :return: a scalar update reference.
    """
    return states.at[jnp.arange(states.shape[0]),
                     turns.astype('uint8'),
                     jnp.full(states.shape[0], row),
                     jnp.full(states.shape[0], col)]


def action_2d_to_indicator(actions_2d: Union[jnp.ndarray, list],
                           states: jnp.ndarray):
    """
    Converts an array of action 2D indices into their sparse indicator array form.

    :param actions_2d: a list of N 2D action indices. Each element is either pass (None),
    or a tuple of integers representing a row,
    column coordinate.
    :param states: a batch array of N Go games.
    :return: a (N x B x B) sparse array representing indicator actions for each state.
    """
    indicator_actions = jnp.zeros(
        (states.shape[0], states.shape[2], states.shape[3]), dtype=bool)
    for i, action in enumerate(actions_2d):
        if action is None:
            continue
        indicator_actions = indicator_actions.at[i, action[0],
                                                 action[1]].set(True)
    return indicator_actions


def action_1d_to_indicator(actions_1d: jnp.ndarray, nrows: int, ncols: int):
    """
    Converts an array of action 1D indices into their sparse indicator array form.

    :param actions_1d: a list of N 1D action indices. Each element is either pass (None or
    an integer equal to the number of actions),
    an integer.
    :param nrows: number of rows
    :param ncols: number of columns
    :return: a (N x B x B) sparse array representing indicator actions for each state.
    """
    batch_size = len(actions_1d)
    indicator_actions = jnp.zeros((batch_size, nrows * ncols + 1), dtype=bool)
    indicator_actions = indicator_actions.at[jnp.arange(batch_size),
                                             actions_1d].set(True)
    return jnp.reshape(indicator_actions[:, :-1], (batch_size, nrows, ncols))


def action_indicator_to_1d(indicator_actions: jnp.ndarray):
    """
    Converts an array of indicator actions to their corresponding action indices.

    :param indicator_actions: n (N x B x B) sparse array. If the values are present, the action
    is a pass.
    :return: an integer array of length N.
    """
    passes = ~jnp.sum(indicator_actions, axis=(1, 2), dtype=bool)
    one_hot_actions = jnp.concatenate((jnp.reshape(
        indicator_actions,
        (len(indicator_actions), -1)), jnp.expand_dims(passes, 1)),
                                      axis=1)
    return jnp.argmax(one_hot_actions, axis=1)
