"""Entry point of the MuZero algorithm for Go."""

import jax
from absl import app, flags

from muzero_gojax import drive, game, logger, manager, metrics, models

_RNG = flags.DEFINE_integer('rng', 42, 'Random seed.')
_BOARD_SIZE = flags.DEFINE_integer("board_size", 5,
                                   "Size of the board for Go games.")
_SKIP_PLAY = flags.DEFINE_bool(
    'skip_play', False,
    'Whether or not to skip playing with the model after training.')
_PLAY_MODEL_SAMPLE_SIZE = flags.DEFINE_integer(
    'play_model_sample_size', 2,
    'Number of times to sample the model when playing against it.')
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


def main(_):
    """
    Main entry of code.
    """
    logger.initialize_start_time()
    drive.initialize_drive(_SAVE_DIR.value)

    # Make model.
    if _LOAD_DIR.value:
        logger.log(f'Loading model from {_LOAD_DIR.value}')
        go_model, params, model_build_config = models.load_model(
            _LOAD_DIR.value)
    else:
        logger.log("Making model from scratch...")
        model_build_config = models.get_model_build_config(_BOARD_SIZE.value)
        go_model, params = models.build_model_with_params(
            model_build_config, jax.random.PRNGKey(_RNG.value))
    metrics.print_param_size_analysis(params)

    # Train model.
    logger.log("Training model...")
    params, metrics_df = manager.train_model(go_model, params,
                                             model_build_config,
                                             jax.random.PRNGKey(_RNG.value),
                                             _SAVE_DIR.value)

    # Metrics.
    if not _SKIP_PLOT.value:
        metrics.plot_all_metrics(go_model, params, metrics_df,
                                 _BOARD_SIZE.value)
    if not _SKIP_ELO_EVAL.value:
        metrics.eval_elo(go_model, params, _BOARD_SIZE.value)
    if not _SKIP_PLAY.value:
        game.play_against_model(
            models.get_policy_model(
                go_model,
                params,
                sample_action_size=_PLAY_MODEL_SAMPLE_SIZE.value),
            _BOARD_SIZE.value,
            play_as_white=_PLAY_AS_WHITE.value,
            rng_key=jax.random.PRNGKey(_RNG.value),
            value_model=models.get_value_model(go_model, params))


if __name__ == '__main__':
    app.run(main)
