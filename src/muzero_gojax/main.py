"""Entry point of the MuZero algorithm for Go."""

import jax
import matplotlib.pyplot as plt
from absl import app
from absl import flags

from muzero_gojax import metrics
from muzero_gojax import models
from muzero_gojax import train

# Training parameters.
flags.DEFINE_integer("batch_size", 2, "Size of the batch to train_model on.")
flags.DEFINE_integer("board_size", 7, "Size of the board for Go games.")
flags.DEFINE_integer("trajectory_length", 50,
                     "Maximum number of game steps for Go. Usually set to 2(board_size^2).")
flags.DEFINE_enum("optimizer", 'sgd', ['sgd', 'adam', 'adamw'], "Optimizer.")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate for the optimizer.")
flags.DEFINE_float("temperature", 0.1, "Temperature for value labels in policy cross entropy loss.")
flags.DEFINE_enum("trans_loss", 'mse', ['mse', 'kl_div', 'bce'], "Transition loss")
flags.DEFINE_bool("add_trans_loss", True,
                  "Whether or not to add the transition loss to the total loss.")
flags.DEFINE_bool("monitor_trans_loss", False,
                  "Whether or not to monitor the transition loss in the plots.")
flags.DEFINE_bool("monitor_trans_acc", False,
                  "Whether or not to monitor the transition accuracy in the plots.")
flags.DEFINE_bool("sigmoid_trans", False,
                  "Apply sigmoid to the transitions when we compute the policy loss and update the "
                  "nt_embeds in update_k_step_losses.")
flags.DEFINE_integer("training_steps", 10, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 0, "How often to evaluate the model.")
flags.DEFINE_integer("rng", 42, "Random seed.")
flags.DEFINE_integer('hypo_steps', '2',
                     'Number of hypothetical steps to take for computing the losses.')

# Model architectures.
flags.DEFINE_integer('hdim', 32, 'Hidden dimension size.')
flags.DEFINE_integer('nlayers', 1, 'Number of layers. Applicable to ResNetV2 models.')
flags.DEFINE_integer('embed_dim', 8, 'Embedded dimension size.')
flags.DEFINE_enum('embed_model', 'black_perspective',
                  ['black_perspective', 'identity', 'linear_conv', 'cnn_lite', 'black_cnn_lite',
                   'black_cnn_medium', 'cnn_medium', 'resnet'],
                  'State embedding model architecture.')
flags.DEFINE_enum('decode_model', 'noop', ['noop', 'resnet'], 'State decoding model architecture.')
flags.DEFINE_enum('value_model', 'linear',
                  ['random', 'linear', 'linear_conv', 'cnn_lite', 'resnet_medium', 'tromp_taylor'],
                  'Value model architecture.')
flags.DEFINE_enum('policy_model', 'linear',
                  ['random', 'linear', 'linear_conv', 'cnn_lite', 'resnet_medium', 'tromp_taylor'],
                  'Policy model architecture.')
flags.DEFINE_enum('transition_model', 'black_perspective',
                  ['real', 'black_perspective', 'random', 'linear_conv', 'cnn_lite', 'cnn_medium',
                   'resnet_medium', 'resnet'], 'Transition model architecture.')

# Serialization.
flags.DEFINE_string('save_dir', None, 'File directory to save the parameters.')
flags.DEFINE_string('load_dir', None,
                    'File path to load the saved parameters. Otherwise the model starts from '
                    'randomly initialized weights.')

# Other.
flags.DEFINE_bool('use_jit', False, 'Use JIT compilation.')
flags.DEFINE_bool('skip_play', False,
                  'Whether or not to skip playing with the model after training.')
flags.DEFINE_bool('skip_plot', False, 'Whether or not to skip plotting anything.')
flags.DEFINE_bool('train_debug_print', False, 'Log stages in the train step function?')

FLAGS = flags.FLAGS


def run(absl_flags: flags.FlagValues):
    """
    Main entry of code.
    """
    print("Making model...")
    go_model = models.make_model(absl_flags)
    print("Initializing model...")
    params = train.init_model(go_model, absl_flags)
    print(f'{sum(x.size for x in jax.tree_util.tree_leaves(params))} parameters.')
    # Plots metrics before training.
    if not absl_flags.skip_plot:
        metrics.plot_histogram_weights(params)
        metrics.plot_model_thoughts(go_model, params,
                                    metrics.get_interesting_states(absl_flags.board_size))
        plt.show()
    print("Training model...")
    params, metrics_df = train.train_model(go_model, params, absl_flags)
    print("Training complete!")
    train.maybe_save_model(params, absl_flags)
    # Plots training results and metrics after training.
    if not absl_flags.skip_plot:
        metrics.plot_metrics(metrics_df)
        metrics.plot_sample_trajectores(absl_flags, go_model, params)
        metrics.plot_histogram_weights(params)
        metrics.plot_model_thoughts(go_model, params,
                                    metrics.get_interesting_states(absl_flags.board_size))
        plt.show()
    if not absl_flags.skip_play:
        metrics.play_against_model(go_model, params, absl_flags)


def main(_):
    """Main function."""
    run(FLAGS)


if __name__ == '__main__':
    app.run(main)
