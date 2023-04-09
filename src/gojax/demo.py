"""Demo usage of `gojax`."""

from jax import numpy as jnp

import gojax

if __name__ == '__main__':
    state = gojax.new_states(7)
    while not jnp.alltrue(state[0, gojax.END_CHANNEL_INDEX]):
        print(gojax.get_string(state[0]))
        USER_INPUT = input('row col: ').strip()
        # pylint: disable=invalid-name
        action = None
        if USER_INPUT:
            row, col = USER_INPUT.split()
            action = (int(row), int(col))
        state = gojax.next_states_legacy(
            state, gojax.action_2d_to_indicator([action], state))

    print(gojax.get_string(state[0]))
