"""Entry point of the MuZero algorithm for Go."""
import jax.nn
from absl import app
from absl import flags

import models
import train

# Training parameters
flags.DEFINE_integer("batch_size", 2, "Size of the batch to train on.")
flags.DEFINE_integer("board_size", 7, "Size of the board for Go games.")
flags.DEFINE_integer("max_num_steps", 50,
                     "Maximum number of game steps for Go. Usually set to 2(board_size^2).")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 10, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 0, "How often to evaluate the model.")
flags.DEFINE_integer("random_seed", 42, "Random seed.")

# Model architectures
flags.DEFINE_enum('embed_model', 'black_perspective',
                  ['black_perspective', 'identity', 'linear', 'black_cnn_lite'],
                  'State embedding model architecture.')
flags.DEFINE_enum('value_model', 'linear', ['random', 'linear'], 'Transition model architecture.')
flags.DEFINE_enum('policy_model', 'linear', ['random', 'linear', 'cnn_lite'],
                  'Policy model architecture.')
flags.DEFINE_enum('transition_model', 'black_perspective',
                  ['real', 'black_perspective', 'random', 'linear', 'cnn_lite'],
                  'Transition model architecture.')

flags.DEFINE_bool('use_jit', False, 'Use JIT compilation.')

FLAGS = flags.FLAGS


def main(_):
    """Program entry point and highest-level algorithm flow of MuZero Go."""
    go_model = models.make_model(FLAGS.board_size, FLAGS.embed_model, FLAGS.value_model,
                                 FLAGS.policy_model,
                                 FLAGS.transition_model)

    rng_key = jax.random.PRNGKey(FLAGS.random_seed)
    _ = train.train(go_model, FLAGS.batch_size, FLAGS.board_size, FLAGS.training_steps,
                    FLAGS.max_num_steps, FLAGS.learning_rate, rng_key, FLAGS.use_jit)

    # TODO: Save the parameters in a specified flag directory defaulted to /tmp.


if __name__ == '__main__':
    app.run(main)
