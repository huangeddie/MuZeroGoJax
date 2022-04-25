import haiku as hk
import jax.nn
from gojax import go

from simulation import self_play
from simulation import trajectories_to_dataset


class RandomGoModel(hk.Module):
    def __call__(self, x):
        return jax.random.normal(hk.next_rng_key(), (x.shape[0], x.shape[2], x.shape[3]))


def update_params(params, trajectories):
    states, state_labels = trajectories_to_dataset(trajectories)
    return params


def train(model_fn, batch_size, board_size, epochs, rng_key):
    params = model_fn.init(rng_key, go.new_states(board_size, 1))
    max_num_steps = 2 * (board_size ** 2)
    for _ in range(epochs):
        trajectories = self_play(model_fn, params, batch_size, board_size, max_num_steps, rng_key)
        params = update_params(params, trajectories)

    return params


def main():
    go_model = hk.transform(lambda states: RandomGoModel()(states))

    board_size = 3
    max_num_steps = 2 * (board_size ** 2)
    batch_size = 1
    epochs = 1
    rng_key = jax.random.PRNGKey(42)
    parameters = train(go_model, batch_size, board_size, epochs, rng_key)

    single_batch_size = 1
    trajectories = self_play(go_model, parameters, single_batch_size, board_size, max_num_steps, rng_key)

    for step in range(trajectories.shape[1]):
        print(f'Step {step}')
        print(go.get_pretty_string(trajectories[0, step]))


if __name__ == '__main__':
    main()
