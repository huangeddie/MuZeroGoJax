"""High-level model management."""

import haiku as hk
import jax.numpy as jnp

from models import policy
from models import state_embed
from models import transition
from models import value


class MockModel(hk.Module):
    """
    Mock model to control the output.

    Assumes the first dimension is the batch size, and repeats the output the same number of
    times as the batch size.
    """

    def __init__(self, output, *args, **kwargs):
        """Initialize with the output to be returned everytime the model is called."""
        super().__init__(*args, **kwargs)
        self.output = output

    def __call__(self, x):
        return jnp.repeat(jnp.array([self.output]), len(x), axis=0)


def make_mock_model(state_embed_output, value_output, policy_output,
                    transition_output) -> hk.MultiTransformed:
    """Builds a mock model that consistently outputs the given arguments."""

    def f():
        # pylint: disable=invalid-name
        state_embed_model = MockModel(state_embed_output)
        value_model = MockModel(value_output)
        policy_model = MockModel(policy_output)
        transition_model = MockModel(transition_output)

        def init(states):
            state_embedding = state_embed_model(states)
            policy_logits = policy_model(state_embedding)
            transition_logits = transition_model(state_embedding)
            value_logits = value_model(state_embedding)
            return value_logits, policy_logits, transition_logits

        return init, (state_embed_model, value_model, policy_model, transition_model)

    return hk.multi_transform(f)


def make_model(board_size: int, state_embed_model_name: str, value_model_name: str,
               policy_model_name: str,
               transition_model_name: str) -> hk.MultiTransformed:
    """
    Builds the corresponding model for the given name.

    :return: A Haiku multi-transformed Go model consisting of (1) a state embedding model,
    (2) a policy model, (3) a transition model, and (4) a value model.
    """

    def f():
        # pylint: disable=invalid-name
        state_embed_model = {'identity': state_embed.StateIdentity}[state_embed_model_name](
            board_size)
        policy_model = {'random': policy.RandomPolicy, 'linear': policy.Linear3DPolicy}[
            policy_model_name](board_size)
        transition_model = {'real': transition.RealTransition,
                            'random': transition.RandomTransition,
                            'linear': transition.Linear3DTransition}[transition_model_name](
            board_size)
        value_model = {'random': value.RandomValue,
                       'linear': value.Linear3DValue}[value_model_name](board_size)

        def init(states):
            state_embedding = state_embed_model(states)
            policy_logits = policy_model(state_embedding)
            transition_logits = transition_model(state_embedding)
            value_logits = value_model(state_embedding)
            return value_logits, policy_logits, transition_logits

        return init, (state_embed_model, value_model, policy_model, transition_model)

    return hk.multi_transform(f)
