"""Entry point of the MuZero algorithm for Go."""
import functools
import os

import haiku as hk
import matplotlib.pyplot as plt
from absl import app
from absl import flags

from muzero_gojax import game
from muzero_gojax import metrics
from muzero_gojax import models
from muzero_gojax import train

_BOARD_SIZE = flags.DEFINE_integer("board_size", 5,
                                   "Size of the board for Go games.")
_SKIP_PLAY = flags.DEFINE_bool(
    'skip_play', False,
    'Whether or not to skip playing with the model after training.')
_SKIP_PLOT = flags.DEFINE_bool('skip_plot', False,
                               'Whether or not to skip plotting anything.')
_SAVE_DIR = flags.DEFINE_string('save_dir', '/tmp/',
                                'File directory to save the parameters.')

FLAGS = flags.FLAGS


def _print_param_size_analysis(params):
    print(f'{hk.data_structures.tree_size(params)} parameters.')

    def _regex_in_dict_item(regex: str, item: tuple):
        return regex in item[0]

    for sub_model_regex in [
            'embed', 'decode', 'value', 'policy', 'transition'
    ]:
        sub_model_params = dict(
            filter(functools.partial(_regex_in_dict_item, sub_model_regex),
                   params.items()))
        print(
            f'\t{sub_model_regex}: {hk.data_structures.tree_size(sub_model_params)} parameters.'
        )


def run(absl_flags: flags.FlagValues):
    """
    Main entry of code.
    """
    print("Making model...")
    go_model, params = models.build_model(_BOARD_SIZE.value)
    _print_param_size_analysis(params)
    # Plots metrics before training.
    if not _SKIP_PLOT.value:
        metrics.plot_model_thoughts(
            go_model, params,
            metrics.get_interesting_states(_BOARD_SIZE.value))
        plt.show()
    print("Training model...")
    params, metrics_df = train.train_model(go_model, params, _BOARD_SIZE.value)
    models.save_model(
        params,
        os.path.join(_SAVE_DIR.value, train.hash_model_flags(absl_flags)))
    if not _SKIP_PLOT.value:
        metrics.plot_metrics(metrics_df)
        metrics.plot_sample_trajectories(
            game.new_trajectories(_BOARD_SIZE.value,
                                  batch_size=2,
                                  trajectory_length=10), go_model, params)
        metrics.plot_model_thoughts(
            go_model, params,
            metrics.get_interesting_states(_BOARD_SIZE.value))
        plt.show()
    if not _SKIP_PLAY.value:
        metrics.play_against_model(go_model, params, _BOARD_SIZE.value)


def main(_):
    """Main function."""
    run(FLAGS)


if __name__ == '__main__':
    app.run(main)
