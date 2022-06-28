"""Entry point of the MuZero algorithm for Go."""
import pickle
import re

import gojax
import jax.random
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
flags.DEFINE_enum('embed_model', 'black_perspective',
                  ['black_perspective', 'identity', 'linear', 'black_cnn_lite',
                   'black_cnn_intermediate'], 'State embedding model architecture.')
flags.DEFINE_enum('value_model', 'linear', ['random', 'linear'], 'Transition model architecture.')
flags.DEFINE_enum('policy_model', 'linear', ['random', 'linear', 'cnn_lite'],
                  'Policy model architecture.')
flags.DEFINE_enum('transition_model', 'black_perspective',
                  ['real', 'black_perspective', 'random', 'linear', 'cnn_lite', 'cnn_intermediate'],
                  'Transition model architecture.')

flags.DEFINE_string('save_path', None, 'File path to save the parameters.')
flags.DEFINE_string('load_path', None,
                    'File path to load the saved parameters. Otherwise the model starts from '
                    'randomly initialized weights.')

flags.DEFINE_bool('use_jit', False, 'Use JIT compilation.')
flags.DEFINE_bool('skip_play', False,
                  'Whether or not to skip playing with the model after training.')

FLAGS = flags.FLAGS

CAP_LETTERS = 'ABCDEFGHIJKLMNOPQRS'


def play(go_model, params, absl_flags):
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


def main(_):
    go_model, params = train.train_from_flags(FLAGS)
    if FLAGS.save_path:
        with open(FLAGS.save_path, 'wb') as f:
            pickle.dump(params, f)
        print(f"Saved model to '{FLAGS.save_path}'.")
    if not FLAGS.skip_play:
        play(go_model, params, FLAGS)


if __name__ == '__main__':
    app.run(main)
