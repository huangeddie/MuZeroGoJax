"""High-level model management."""

import haiku as hk

from models import embed
from models import policy
from models import transition
from models import value


def make_model(board_size: int, embed_model_name: str, value_model_name: str,
               policy_model_name: str,
               transition_model_name: str) -> hk.MultiTransformed:
    """
    Builds the corresponding model for the given name.

    :return: A Haiku multi-transformed Go model consisting of (1) a state embedding model,
    (2) a policy model, (3) a transition model, and (4) a value model.
    """

    def f():
        # pylint: disable=invalid-name
        embed_model = {'identity': embed.Identity,
                       'black_perspective': embed.BlackPerspective,
                       'black_cnn_lite': embed.BlackCNNLite}[embed_model_name](board_size)
        value_model = {'random': value.RandomValue,
                       'linear': value.Linear3DValue}[value_model_name](board_size)
        policy_model = {'random': policy.RandomPolicy,
                        'linear': policy.Linear3DPolicy,
                        'cnn_lite': policy.CNNLitePolicy}[policy_model_name](board_size)
        transition_model = {'real': transition.RealTransition,
                            'black_perspective': transition.BlackRealTransition,
                            'random': transition.RandomTransition,
                            'linear': transition.Linear3DTransition,
                            'cnn_lite': transition.CNNLiteTransition}[transition_model_name](
            board_size)

        def init(states):
            embedding = embed_model(states)
            policy_logits = policy_model(embedding)
            transition_logits = transition_model(embedding)
            value_logits = value_model(embedding)
            return value_logits, policy_logits, transition_logits

        return init, (embed_model, value_model, policy_model, transition_model)

    return hk.multi_transform(f)
