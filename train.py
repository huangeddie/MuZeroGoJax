import gojax

from game import trajectories_to_dataset, self_play


def update_params(params, trajectories):
    states, state_labels = trajectories_to_dataset(trajectories)
    return params


def train(model_fn, batch_size, board_size, training_steps, max_num_steps, rng_key):
    params = model_fn.init(rng_key, gojax.new_states(board_size, 1))
    for _ in range(training_steps):
        trajectories = self_play(model_fn, params, batch_size, board_size, max_num_steps, rng_key)
        params = update_params(params, trajectories)

    return params