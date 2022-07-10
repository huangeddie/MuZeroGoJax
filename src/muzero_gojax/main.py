"""Entry point of the MuZero algorithm for Go."""
import os.path
import pickle
import re

import gojax
import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
from absl import app
from absl import flags

from muzero_gojax import game
from muzero_gojax import train

# Training parameters
flags.DEFINE_integer("batch_size", 2, "Size of the batch to train_model on.")
flags.DEFINE_integer("board_size", 7, "Size of the board for Go games.")
flags.DEFINE_integer("max_num_steps", 50,
                     "Maximum number of game steps for Go. Usually set to 2(board_size^2).")
flags.DEFINE_enum("optimizer", 'sgd', ['sgd', 'adam'], "Optimizer.")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 10, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 0, "How often to evaluate the model.")
flags.DEFINE_integer("random_seed", 42, "Random seed.")

# Model architectures
flags.DEFINE_integer('hdim', '32', 'Hidden dimension size.')
flags.DEFINE_enum('embed_model', 'black_perspective',
                  ['black_perspective', 'identity', 'linear', 'black_cnn_lite',
                   'black_cnn_intermediate'], 'State embedding model architecture.')
flags.DEFINE_enum('value_model', 'linear', ['random', 'linear'], 'Transition model architecture.')
flags.DEFINE_enum('policy_model', 'linear', ['random', 'linear', 'cnn_lite'],
                  'Policy model architecture.')
flags.DEFINE_enum('transition_model', 'black_perspective',
                  ['real', 'black_perspective', 'random', 'linear', 'cnn_lite', 'cnn_intermediate'],
                  'Transition model architecture.')

flags.DEFINE_string('save_dir', None, 'File directory to save the parameters.')
flags.DEFINE_string('load_path', None,
                    'File path to load the saved parameters. Otherwise the model starts from '
                    'randomly initialized weights.')

flags.DEFINE_bool('use_jit', False, 'Use JIT compilation.')
flags.DEFINE_bool('skip_play', False,
                  'Whether or not to skip playing with the model after training.')
flags.DEFINE_bool('skip_policy_plot', False,
                  'Whether or not to skip plotting the policy of the model.')

FLAGS = flags.FLAGS

CAP_LETTERS = 'ABCDEFGHIJKLMNOPQRS'


def play_against_model(go_model, params, absl_flags):
    """
    Deploys an interactive terminal to play against the Go model.

    :param go_model: Haiku Go model.
    :param params: Model parameters.
    :param absl_flags: ABSL flags.
    :return: None.
    """
    states = gojax.new_states(absl_flags.board_size)
    print(gojax.get_pretty_string(states[0]))
    rng_key = jax.random.PRNGKey(absl_flags.random_seed)
    step = 0
    while not gojax.get_ended(states):
        # Get user's move.
        re_match = re.match('\s*(\d+)\s+(\D+)\s*', input('Enter move (R C):'))
        while not re_match:
            re_match = re.match('\s*(\d+)\s+(\D+)\s*', input('Enter move (R C):'))
        row = int(re_match.group(1))
        col = CAP_LETTERS.index(re_match.group(2).upper())
        indicator_actions = gojax.action_2d_indices_to_indicator([(row, col)], states)
        states = gojax.next_states(states, indicator_actions)
        print(gojax.get_pretty_string(states[0]))
        if gojax.get_ended(states):
            break

        # Get AI's move.
        print('Model thinking...')
        rng_key = jax.random.fold_in(rng_key, step)
        states = game.sample_next_states(go_model, params, rng_key, states)
        print(gojax.get_pretty_string(states[0]))
        step += 1


def plot_policy_heat_map(go_model, params, state, rng_key=None):
    """
    Plots a heatmap of the policy for the given state.

    Plots (1) the state, (2) the non-pass action logits, (3) the pass logit.
    """
    if not rng_key:
        rng_key = jax.random.PRNGKey(42)
    logits = game.get_policy_logits(go_model, params, jnp.expand_dims(state, 0), rng_key)
    action_logits, pass_logit = logits[0, :-1], logits[0, -1]
    action_logits = jnp.reshape(action_logits, state.shape[1:])
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title('State')
    plt.imshow(
        state[gojax.BLACK_CHANNEL_INDEX].astype(int) - state[gojax.WHITE_CHANNEL_INDEX].astype(int),
        vmin=-1, vmax=1, cmap='Greys')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title('Action logits')
    plt.imshow(action_logits, vmin=-3, vmax=3)
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title('Pass logit')
    plt.bar([0], [pass_logit])
    plt.ylim(-3, 3)
    plt.show()


def maybe_save_model(params, absl_flags):
    """
    Saves the parameters with a filename that is the hash of the absl_flags

    :param params: Dictionary of parameters.
    :param absl_flags: ABSL flags.
    :return: None.
    """
    if absl_flags.save_dir:
        filename = os.path.join(absl_flags.save_dir,
                                str(hash(absl_flags.flags_into_string())) + '.pickle')
        with open(filename, 'wb') as f:
            pickle.dump(params, f)
        print(f"Saved model to '{filename}'.")
        return filename
    else:
        print(f"Model NOT saved.")


def run(absl_flags):
    """
    Main entry of code.
    """
    go_model, params = train.train_from_flags(absl_flags)
    maybe_save_model(params, absl_flags)
    if not absl_flags.skip_policy_plot:
        plot_policy_heat_map(go_model, params, gojax.new_states(absl_flags.board_size)[0])
    if not absl_flags.skip_play:
        play_against_model(go_model, params, absl_flags)


def main(_):
    run(FLAGS)


if __name__ == '__main__':
    app.run(main)
