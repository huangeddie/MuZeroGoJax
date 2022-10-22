"""Entry point of the MuZero algorithm for Go."""

import os

import jax
import matplotlib.pyplot as plt
from absl import app
from absl import flags

from muzero_gojax import game
from muzero_gojax import metrics
from muzero_gojax import models
from muzero_gojax import train

_BOARD_SIZE = flags.DEFINE_integer("board_size", 7, "Size of the board for Go games.")
_SKIP_PLAY = flags.DEFINE_bool('skip_play', False,
                               'Whether or not to skip playing with the model after training.')
_SKIP_PLOT = flags.DEFINE_bool('skip_plot', False, 'Whether or not to skip plotting anything.')
_SAVE_DIR = flags.DEFINE_string('save_dir', './', 'File directory to save the parameters.')

FLAGS = flags.FLAGS


def run(absl_flags: flags.FlagValues):
    """
    Main entry of code.
    """
    print("Making model...")
    go_model, params = models.make_model(_BOARD_SIZE.value)
    print(f'{sum(x.size for x in jax.tree_util.tree_leaves(params))} parameters.')
    # Plots metrics before training.
    if not _SKIP_PLOT.value:
        metrics.plot_histogram_weights(params)
        metrics.plot_model_thoughts(go_model, params,
                                    metrics.get_interesting_states(_BOARD_SIZE.value))
        plt.show()
    print("Training model...")
    params, metrics_df = train.train_model(go_model, params, _BOARD_SIZE.value)
    print("Training complete!")
    models.save_model(params, os.path.join(_SAVE_DIR.value, train.hash_model_flags(absl_flags)))
    if not _SKIP_PLOT.value:
        metrics.plot_metrics(metrics_df)
        metrics.plot_sample_trajectories(
            game.new_trajectories(_BOARD_SIZE.value, batch_size=2, trajectory_length=10), go_model,
            params)
        metrics.plot_histogram_weights(params)
        metrics.plot_model_thoughts(go_model, params,
                                    metrics.get_interesting_states(_BOARD_SIZE.value))
        plt.show()
    if not _SKIP_PLAY.value:
        metrics.play_against_model(go_model, params, _BOARD_SIZE.value)


def main(_):
    """Main function."""
    run(FLAGS)


if __name__ == '__main__':
    app.run(main)
