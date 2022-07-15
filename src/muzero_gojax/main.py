"""Entry point of the MuZero algorithm for Go."""

import absl.flags
import gojax
from absl import app
from absl import flags

# Training parameters.
from muzero_gojax import metrics
from muzero_gojax import models
from muzero_gojax import train

flags.DEFINE_integer("batch_size", 2, "Size of the batch to train_model on.")
flags.DEFINE_integer("board_size", 7, "Size of the board for Go games.")
flags.DEFINE_integer("max_num_steps", 50,
                     "Maximum number of game steps for Go. Usually set to 2(board_size^2).")
flags.DEFINE_enum("optimizer", 'sgd', ['sgd', 'adam', 'adamw'], "Optimizer.")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 10, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 0, "How often to evaluate the model.")
flags.DEFINE_integer("random_seed", 42, "Random seed.")

# Model architectures.
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

# Serialization.
flags.DEFINE_string('save_dir', None, 'File directory to save the parameters.')
flags.DEFINE_string('load_path', None,
                    'File path to load the saved parameters. Otherwise the model starts from '
                    'randomly initialized weights.')

# Other.
flags.DEFINE_bool('use_jit', False, 'Use JIT compilation.')
flags.DEFINE_bool('skip_play', False,
                  'Whether or not to skip playing with the model after training.')
flags.DEFINE_bool('skip_policy_plot', False,
                  'Whether or not to skip plotting the policy of the model.')

FLAGS = flags.FLAGS


def run(absl_flags: absl.flags.FlagValues):
    """
    Main entry of code.
    """
    print("Making model...")
    go_model = models.make_model(absl_flags)
    print("Initializing model...")
    params = train.init_model(go_model, absl_flags)
    print("Training model...")
    params, metrics_df = train.train_model(go_model, params, absl_flags)
    print("Training complete!")
    train.maybe_save_model(params, absl_flags)
    metrics.plot_metrics(metrics_df)
    if not absl_flags.skip_policy_plot:
        metrics.plot_policy_heat_map(go_model, params, gojax.new_states(absl_flags.board_size)[0])
    if not absl_flags.skip_play:
        metrics.play_against_model(go_model, params, absl_flags)


def main(_):
    run(FLAGS)


if __name__ == '__main__':
    app.run(main)
