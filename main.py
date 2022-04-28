import gojax
import haiku as hk
import jax.nn
from absl import app
from absl import flags

from game import self_play
from game import trajectories_to_dataset

flags.DEFINE_integer("batch_size", 1, "Size of the batch to train on.")
flags.DEFINE_integer("board_size", 3, "Size of the board for Go games.")
flags.DEFINE_integer("max_num_steps", 18, "Maximum number of game steps for Go. Usually set to 2(board_size^2).")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 1, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 100, "How often to evaluate the model.")
flags.DEFINE_integer("random_seed", 42, "Random seed.")
flags.DEFINE_enum('model_class_name', 'random', ['random', 'linear', 'conv_linear_1x1', 'conv_linear_3x3'],
                  'Model architecture class.')

FLAGS = flags.FLAGS


class RandomGoModel(hk.Module):
    def __call__(self, x):
        return jax.random.normal(hk.next_rng_key(), (x.shape[0], x.shape[2] * x.shape[3] + 1))


def get_model(model_class: str) -> hk.Transformed:
    return hk.transform(lambda states: {'random': RandomGoModel}[model_class]()(states))


def update_params(params, trajectories):
    states, state_labels = trajectories_to_dataset(trajectories)
    return params


def train(model_fn, batch_size, board_size, training_steps, max_num_steps, rng_key):
    params = model_fn.init(rng_key, gojax.new_states(board_size, 1))
    for _ in range(training_steps):
        trajectories = self_play(model_fn, params, batch_size, board_size, max_num_steps, rng_key)
        params = update_params(params, trajectories)

    return params


def main(_):
    go_model = get_model(FLAGS.model_class_name)

    rng_key = jax.random.PRNGKey(FLAGS.random_seed)
    parameters = train(go_model, FLAGS.batch_size, FLAGS.board_size, FLAGS.training_steps, FLAGS.max_num_steps, rng_key)

    single_batch_size = 1
    trajectories = self_play(go_model, parameters, single_batch_size, FLAGS.board_size, FLAGS.max_num_steps, rng_key)

    for step in range(trajectories.shape[1]):
        print(f'Step {step}')
        print(gojax.get_pretty_string(trajectories[0, step]))


if __name__ == '__main__':
    app.run(main)
