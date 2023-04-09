"""Encodes and decodes Go states to and from strings."""

import textwrap

from jax import numpy as jnp

import gojax

CAP_LETTERS = 'ABCDEFGHIJKLMNOPQRS'


def _decode_single_state(encode_str, ended, passed, turn):
    lines = encode_str.splitlines()
    board_size = len(lines[0].split())
    states = gojax.new_states(board_size, batch_size=1)
    for i, line in enumerate(lines):
        for j, char in enumerate(line.split()):
            if char == 'B':
                states = states.at[0, gojax.BLACK_CHANNEL_INDEX, i,
                                   j].set(True)
            elif char == 'W':
                states = states.at[0, gojax.WHITE_CHANNEL_INDEX, i,
                                   j].set(True)
            elif char == 'X':
                states = states.at[0, gojax.KILLED_CHANNEL_INDEX, i,
                                   j].set(True)
        if i == board_size:
            # Extract metadata.
            for key_value in line.split(';'):
                key, value = key_value.split('=')
                value = value.upper()
                if key == 'TURN':
                    if value in ['1', 'W', 'WHITE', 'T', 'TRUE']:
                        turn = True
                    elif value not in ['0', 'B', 'BLACK', 'F', 'FALSE']:
                        raise ValueError(f'Invalid TURN value: {value}')
                elif key == 'PASS':
                    if value in ['1', 'T', 'TRUE']:
                        passed = True
                    elif value not in ['0', 'F', 'FALSE']:
                        raise ValueError(f'Invalid PASS value: {value}')
                elif key == 'END':
                    if value in ['1', 'T', 'TRUE']:
                        ended = True
                    elif value not in ['0', 'F', 'FALSE']:
                        raise ValueError(f'Invalid END value: {value}')
                else:
                    raise ValueError(f'Unknown macro: {key}')

    # Set the turn.
    states = states.at[0, gojax.TURN_CHANNEL_INDEX].set(turn)
    # Set passed.
    states = states.at[0, gojax.PASS_CHANNEL_INDEX].set(passed)
    # Set ended.
    states = states.at[0, gojax.END_CHANNEL_INDEX].set(ended)

    return states


def decode_states(serialized_states: str,
                  turn: bool = gojax.BLACKS_TURN,
                  passed: bool = False,
                  ended: bool = False):
    """
    Creates game boards from a human-readable serialzied string.

    Each state in the encoding is assumed to be separated by 2 consecutive new lines.

    Example encodings:
    ```
    B W
    W _

    B W
    W X
    TURN=W;PASS=TRUE;END=FALSE
    ```

    :param serialized_states: string representations of the Go games.
    :param turn: boolean turn indicator.
    :param passed: boolean indicator if the previous move was passed.
    :param ended: whether the game ended.
    :return: a N x C x B X B boolean array.
    """
    if serialized_states[0] == '\n':
        serialized_states = serialized_states[1:]
    if serialized_states[-1] == '\n':
        serialized_states = serialized_states[:-1]
    serialized_states = textwrap.dedent(serialized_states)
    states = []
    for serialized_state in serialized_states.split('\n\n'):
        states.append(
            _decode_single_state(serialized_state, ended, passed, turn))

    return jnp.concatenate(states)


def _get_second_character_go_pretty_string(i, j, size):
    """Returns the corresponding second character for the given position on the Go board."""
    if i == 0:
        if j == 0:
            character = '╔'
        elif j == size - 1:
            character = '╗'
        else:
            character = '╤'
    elif i == size - 1:
        if j == 0:
            character = '╚'
        elif j == size - 1:
            character = '╝'
        else:
            character = '╧'
    else:
        if j == 0:
            character = '╟'
        elif j == size - 1:
            character = '╢'
        else:
            character = '┼'
    return character


def get_string(state):
    """
    Creates a human-friendly string of the given state.

    :param state: (1 x) C x B x B boolean array.
    :return: string representing the state.
    """
    if jnp.ndim(state) == 4 and state.shape[0] == 1:
        state = jnp.squeeze(state, axis=0)
    board_str = ''
    size = state.shape[1]
    board_str += '\t'
    for i in range(size):
        board_str += f'{CAP_LETTERS[i]}'.ljust(2, ' ')
    board_str += '\n'
    for i in range(size):
        board_str += f'{i}\t'
        for j in range(size):
            # First character
            if state[gojax.BLACK_CHANNEL_INDEX, i, j]:
                board_str += '●'
            elif state[gojax.WHITE_CHANNEL_INDEX, i, j]:
                board_str += '○'
            elif state[gojax.KILLED_CHANNEL_INDEX, i, j]:
                board_str += 'x'
            else:
                board_str += _get_second_character_go_pretty_string(i, j, size)

            # Second character
            if j != size - 1:
                if i in (0, size - 1):
                    board_str += '═'
                else:
                    board_str += '─'
        board_str += '\n'

    areas = gojax.compute_area_sizes(jnp.expand_dims(state, 0))
    done = jnp.alltrue(state[gojax.END_CHANNEL_INDEX])
    previous_player_passed = jnp.alltrue(state[gojax.PASS_CHANNEL_INDEX])
    turn = jnp.alltrue(state[gojax.TURN_CHANNEL_INDEX])
    if done:
        game_state = 'END'
    elif previous_player_passed:
        game_state = 'PASSED'
    else:
        game_state = 'ONGOING'
    board_str += f"\tTurn: {'BLACK' if turn == 0 else 'WHITE'}, Game State: {game_state}\n"
    board_str += f'\tBlack Area: {areas[0, 0]}, White Area: {areas[0, 1]}\n'
    return board_str


def _encode_single_state(state: jnp.ndarray) -> str:
    if jnp.ndim(state) == 4 and state.shape[0] == 1:
        state = jnp.squeeze(state, axis=0)
    board_str = ''
    size = state.shape[1]
    for i in range(size):
        for j in range(size):
            if state[gojax.BLACK_CHANNEL_INDEX, i, j]:
                board_str += 'B'
            elif state[gojax.WHITE_CHANNEL_INDEX, i, j]:
                board_str += 'W'
            elif state[gojax.KILLED_CHANNEL_INDEX, i, j]:
                board_str += 'X'
            else:
                board_str += '_'
            board_str += ' '
        board_str += '\n'

    done = jnp.alltrue(state[gojax.END_CHANNEL_INDEX])
    previous_player_passed = jnp.alltrue(state[gojax.PASS_CHANNEL_INDEX])
    turn = jnp.alltrue(state[gojax.TURN_CHANNEL_INDEX])
    macros = []
    if turn:
        macros.append('TURN=W')
    if previous_player_passed:
        macros.append('PASS=T')
    if done:
        macros.append('END=T')
    board_str += ';'.join(macros)
    if len(macros):
        board_str += '\n'
    return board_str


def encode_states(states: jnp.ndarray) -> str:
    """
    Encodes a batch of states into a string encoding.

    NOTE: Does not include KOMI / KILLED pieces.

    :param states:
    :return: Encoded string of states.
    """
    return '\n'.join(map(lambda state: _encode_single_state(state), states))


def print_state(state):
    """
    Prints a human-friendly string of the given state.

    :param state: (1 x) C x B x B boolean array.
    """
    print(get_string(state))
