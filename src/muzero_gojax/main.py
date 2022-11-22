"""Entry point of the MuZero algorithm for Go."""
import functools
import os

import haiku as hk
import jax
import matplotlib.pyplot as plt
from absl import app, flags

from muzero_gojax import game, metrics, models, train

_RNG = flags.DEFINE_integer('rng', 42, 'Random seed.')
_BOARD_SIZE = flags.DEFINE_integer("board_size", 5,
                                   "Size of the board for Go games.")
_DTYPE = flags.DEFINE_enum('dtype', 'float32', ['bfloat16', 'float32'],
                           'Data type.')
_SKIP_PLAY = flags.DEFINE_bool(
    'skip_play', False,
    'Whether or not to skip playing with the model after training.')
_SKIP_PLOT = flags.DEFINE_bool('skip_plot', False,
                               'Whether or not to skip plotting anything.')
_SKIP_ELO_EVAL = flags.DEFINE_bool(
    'skip_elo_eval', False,
    'Skips evaluating the trained model against baseline models.')
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
            f'\t{sub_model_regex}: {hk.data_structures.tree_size(sub_model_params)}'
        )


def main(_):
    """
    Main entry of code.
    """
    rng_key = jax.random.PRNGKey(_RNG.value)
    print("Making model...")
    go_model, params = models.build_model_with_params(_BOARD_SIZE.value,
                                                      _DTYPE.value, rng_key)
    _print_param_size_analysis(params)
    # Plots metrics before training.
    if not _SKIP_PLOT.value:
        metrics.plot_model_thoughts(
            go_model, params,
            metrics.get_interesting_states(_BOARD_SIZE.value))
        plt.show()
    print("Training model...")
    params, metrics_df = train.train_model(go_model, params, _BOARD_SIZE.value,
                                           _DTYPE.value, rng_key)
    models.save_model(
        params, os.path.join(_SAVE_DIR.value, train.hash_model_flags(FLAGS)))
    # Plot metrics after training.
    if not _SKIP_PLOT.value:
        metrics.plot_metrics(metrics_df)
        metrics.plot_sample_trajectories(
            game.new_trajectories(_BOARD_SIZE.value,
                                  batch_size=2,
                                  trajectory_length=_BOARD_SIZE.value**2),
            go_model, params)
        metrics.plot_model_thoughts(
            go_model, params,
            metrics.get_interesting_states(_BOARD_SIZE.value))
        plt.show()
    # Get win rates against benchmark models
    if not _SKIP_ELO_EVAL.value:
        n_games = 256
        random_wins, random_ties, random_losses = game.pit(
            models.get_policy_model(go_model, params),
            models.get_policy_model(models.make_random_model(), params={}),
            _BOARD_SIZE.value,
            n_games,
            traj_len=_BOARD_SIZE.value**2)
        random_win_rate = (random_wins + random_ties / 2) / n_games
        print(f"Against random model: {random_win_rate:.3f} win rate "
              f"| {random_wins} wins, {random_ties} ties, "
              f"{random_losses} losses")
        tromp_taylor_wins, tromp_taylor_ties, tromp_taylor_losses = game.pit(
            models.get_policy_model(go_model, params),
            models.get_policy_model(models.make_tromp_taylor_model(),
                                    params={}),
            _BOARD_SIZE.value,
            n_games,
            traj_len=_BOARD_SIZE.value**2)
        tromp_taylor_win_rate = (tromp_taylor_wins +
                                 tromp_taylor_ties / 2) / n_games
        print(
            f"Against Tromp Taylor model: {tromp_taylor_win_rate:.3f} "
            f"win rate | {tromp_taylor_wins} wins, {tromp_taylor_ties} ties, "
            f"{tromp_taylor_losses} losses")
    # Play against the model.
    if not _SKIP_PLAY.value:
        game.play_against_model(models.get_policy_model(go_model, params),
                                _BOARD_SIZE.value)


if __name__ == '__main__':
    app.run(main)
