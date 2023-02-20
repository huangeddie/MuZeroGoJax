"""Entry point of the MuZero algorithm for Go."""

import haiku as hk
import jax
import matplotlib.pyplot as plt
import optax
import pandas as pd
from absl import app, flags

from muzero_gojax import game, logger, metrics, models, train

_RNG = flags.DEFINE_integer('rng', 42, 'Random seed.')
_BOARD_SIZE = flags.DEFINE_integer("board_size", 5,
                                   "Size of the board for Go games.")
_DTYPE = flags.DEFINE_enum('dtype', 'float32', ['bfloat16', 'float32'],
                           'Data type.')
_SKIP_PLAY = flags.DEFINE_bool(
    'skip_play', False,
    'Whether or not to skip playing with the model after training.')
_PLAY_AS_WHITE = flags.DEFINE_bool(
    'play_as_white', False,
    'Whether or not to skip playing with the model after training.')
_SKIP_PLOT = flags.DEFINE_bool('skip_plot', False,
                               'Whether or not to skip plotting anything.')
_SKIP_ELO_EVAL = flags.DEFINE_bool(
    'skip_elo_eval', True,
    'Skips evaluating the trained model against baseline models.')
_SAVE_DIR = flags.DEFINE_string('save_dir', '/tmp/checkpoint/',
                                'File directory to save the model.')
_LOAD_DIR = flags.DEFINE_string(
    'load_dir', None, 'Directory path to load the model.'
    'Otherwise the model starts from randomly '
    'initialized weights.')

FLAGS = flags.FLAGS


def _plot_all_metrics(go_model: hk.MultiTransformed, params: optax.Params,
                      metrics_df: pd.DataFrame):
    logger.log("Plotting all metrics.")
    if len(metrics_df) > 0:
        metrics.plot_train_metrics_by_regex(metrics_df)
    else:
        logger.log("No training metrics to plot.")
    metrics.plot_sample_trajectories(
        game.new_trajectories(_BOARD_SIZE.value,
                              batch_size=3,
                              trajectory_length=2 * _BOARD_SIZE.value**2),
        go_model, params)
    metrics.plot_sample_trajectories(
        game.new_trajectories(_BOARD_SIZE.value,
                              batch_size=1,
                              trajectory_length=2 * _BOARD_SIZE.value**2),
        go_model,
        params,
        policy_model=models.get_policy_model(models.make_random_model(),
                                             params={}))
    plt.show()


def main(_):
    """
    Main entry of code.
    """
    logger.initialize_start_time()
    rng_key = jax.random.PRNGKey(_RNG.value)
    if _LOAD_DIR.value:
        logger.log(f'Loading model from {_LOAD_DIR.value}')
        go_model, params, all_models_build_config = models.load_model(
            _LOAD_DIR.value)

    else:
        logger.log("Making model from scratch...")
        all_models_build_config = models.get_all_models_build_config(
            _BOARD_SIZE.value, _DTYPE.value)
        go_model, params = models.build_model_with_params(
            all_models_build_config, rng_key)
    metrics.print_param_size_analysis(params)
    logger.log("Training model...")
    params, metrics_df = train.train_model(go_model, params, _BOARD_SIZE.value,
                                           _DTYPE.value, rng_key)
    models.save_model(params, all_models_build_config, _SAVE_DIR.value)
    if not _SKIP_PLOT.value:
        _plot_all_metrics(go_model, params, metrics_df)
    if not _SKIP_ELO_EVAL.value:
        metrics.eval_elo(go_model, params, _BOARD_SIZE.value)
    if not _SKIP_PLAY.value:
        game.play_against_model(models.get_policy_model(go_model, params),
                                _BOARD_SIZE.value,
                                play_as_white=_PLAY_AS_WHITE.value)


if __name__ == '__main__':
    app.run(main)
