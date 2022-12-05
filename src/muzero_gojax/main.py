"""Entry point of the MuZero algorithm for Go."""
import functools

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
_SAVE_DIR = flags.DEFINE_string('save_dir', '/tmp/checkpoint/',
                                'File directory to save the model.')
_LOAD_DIR = flags.DEFINE_string(
    'load_dir', None, 'Directory path to load the model.'
    'Otherwise the model starts from randomly '
    'initialized weights.')

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


def _plot_all_metrics(go_model, params, metrics_df):
    metrics.plot_train_metrics_by_regex(metrics_df)
    metrics.plot_sample_trajectories(
        game.new_trajectories(_BOARD_SIZE.value,
                              batch_size=2,
                              trajectory_length=_BOARD_SIZE.value**2),
        go_model, params)
    metrics.plot_model_thoughts(
        go_model, params, metrics.get_interesting_states(_BOARD_SIZE.value))
    plt.show()


def _eval_elo(go_model, params):
    """Evaluates the ELO by pitting it against baseline models."""
    n_games = 256
    base_policy_model = models.get_policy_model(go_model, params)
    improved_policy_model = models.get_policy_model(go_model,
                                                    params,
                                                    sample_action_size=2)
    for policy_model, policy_name in [(base_policy_model, 'Base'),
                                      (improved_policy_model, 'Improved (2)')]:
        for benchmark in models.get_benchmarks():
            wins, ties, losses = game.pit(policy_model,
                                          benchmark.policy,
                                          _BOARD_SIZE.value,
                                          n_games,
                                          traj_len=_BOARD_SIZE.value**2)
            win_rate = (wins + ties / 2) / n_games
            print(
                f"{policy_name} v. {benchmark.name}: {win_rate:.3f} win rate "
                f"| {wins} wins, {ties} ties, {losses} losses")


def main(_):
    """
    Main entry of code.
    """
    rng_key = jax.random.PRNGKey(_RNG.value)
    if _LOAD_DIR.value:
        print(f'Loading model from {_LOAD_DIR.value}')
        go_model, params, all_models_build_config = models.load_model(
            _LOAD_DIR.value)

    else:
        print("Making model from scratch...")
        all_models_build_config = models.get_all_models_build_config(
            _BOARD_SIZE.value, _DTYPE.value)
        go_model, params = models.build_model_with_params(
            all_models_build_config, rng_key)
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
    models.save_model(params, all_models_build_config, _SAVE_DIR.value)
    if not _SKIP_PLOT.value:
        _plot_all_metrics(go_model, params, metrics_df)
    if not _SKIP_ELO_EVAL.value:
        _eval_elo(go_model, params)
    if not _SKIP_PLAY.value:
        game.play_against_model(models.get_policy_model(go_model, params),
                                _BOARD_SIZE.value)


if __name__ == '__main__':
    app.run(main)
