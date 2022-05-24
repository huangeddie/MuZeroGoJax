"""Entry point of the MuZero algorithm for Go."""
import gojax
import jax.nn
from absl import app
from absl import flags

import game
import models
import train

# Training parameters
flags.DEFINE_integer("batch_size", 1, "Size of the batch to train on.")
flags.DEFINE_integer("board_size", 3, "Size of the board for Go games.")
flags.DEFINE_integer("max_num_steps", 18,
                     "Maximum number of game steps for Go. Usually set to 2(board_size^2).")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 1, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 100, "How often to evaluate the model.")
flags.DEFINE_integer("random_seed", 42, "Random seed.")

# Model architectures
flags.DEFINE_enum('embed_model', 'identity', ['identity', 'linear'],
                  'State embedding model architecture.')
flags.DEFINE_enum('policy_model', 'random', ['random', 'linear'], 'Policy model architecture.')
flags.DEFINE_enum('transition_model', 'real', ['real', 'random', 'linear'],
                  'Transition model architecture.')
flags.DEFINE_enum('value_model', 'random', ['random', 'linear'], 'Transition model architecture.')

FLAGS = flags.FLAGS


def main(_):
    """Program entry point and highest-level algorithm flow of MuZero Go."""
    go_model = models.make_model(FLAGS.board_size, FLAGS.embed_model, FLAGS.policy_model,
                                 FLAGS.transition_model,
                                 FLAGS.value_model)

    rng_key = jax.random.PRNGKey(FLAGS.random_seed)
    params = train.train(go_model, FLAGS.batch_size, FLAGS.board_size, FLAGS.training_steps,
                         FLAGS.max_num_steps, FLAGS.learning_rate,
                         rng_key)

    single_batch_size = 1
    trajectories = game.self_play(go_model, params, single_batch_size, FLAGS.board_size,
                                  FLAGS.max_num_steps, rng_key)

    for step in range(trajectories.shape[1]):
        print(f'Step {step}')
        print(gojax.get_pretty_string(trajectories[0, step]))


if __name__ == '__main__':
    app.run(main)
