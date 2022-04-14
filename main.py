import haiku as hk
from jax import numpy as jnp
from gojax import go
from gojax import constants


class GoModel(hk.Module):
    def __call__(self, x):
        w = hk.get_parameter("w", [], init=jnp.zeros)
        return jnp.sum(x * w, axis=(1,))


def _call_go_model(states):
    return GoModel()(states)


def train(model_fn):
    states = go.new_states(7, 1)
    params = model_fn.init(None, states)
    print(params)
    return params


if __name__ == '__main__':
    call_go_model = hk.without_apply_rng(hk.transform(_call_go_model))
    parameters = train(call_go_model)

    states = go.new_states(7, 1)
    while(not states[0, constants.END_CHANNEL_INDEX, 0, 0]):
        action_logits = call_go_model.apply(parameters, states)
        available_action_logits = action_logits * jnp.array(~states[:, constants.INVALID_CHANNEL_INDEX], dtype=float)


