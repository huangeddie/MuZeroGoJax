"""High-level model management."""

import haiku as hk

from muzero_gojax.models import embed
from muzero_gojax.models import policy
from muzero_gojax.models import transition
from muzero_gojax.models import value


def make_model(absl_flags) -> hk.MultiTransformed:
    """
    Builds the corresponding model for the given name.

    :param absl_flags: Abseil flags.
    :return: A Haiku multi-transformed Go model consisting of (1) a state embedding model,
    (2) a policy model, (3) a transition model, and (4) a value model.
    """
    board_size = absl_flags.board_size
    hdim = absl_flags.hdim

    def f():
        # pylint: disable=invalid-name
        embed_model = \
            {'identity': embed.Identity, 'linear': embed.LinearConvEmbed, 'black_perspective': embed.BlackPerspective,
             'black_cnn_lite': embed.BlackCNNLite, 'black_cnn_intermediate': embed.BlackCNNIntermediate,
             'cnn_lite': embed.CNNLiteEmbed, 'cnn_intermediate': embed.CNNIntermediateEmbed}[absl_flags.embed_model](
                board_size, hdim)
        value_model = \
            {'random': value.RandomValue, 'linear': value.Linear3DValue, 'tromp_taylor': value.TrompTaylorValue}[
                absl_flags.value_model](board_size, hdim)
        policy_model = \
            {'random': policy.RandomPolicy, 'linear': policy.Linear3DPolicy, 'cnn_lite': policy.CNNLitePolicy,
             'tromp_taylor': policy.TrompTaylorPolicy}[absl_flags.policy_model](board_size, hdim)
        transition_model = {'real': transition.RealTransition, 'black_perspective': transition.BlackRealTransition,
                            'random': transition.RandomTransition, 'linear': transition.Linear3DTransition,
                            'cnn_lite': transition.CNNLiteTransition,
                            'cnn_intermediate': transition.CNNIntermediateTransition}[absl_flags.transition_model](
            board_size, hdim)

        def init(states):
            embedding = embed_model(states)
            policy_logits = policy_model(embedding)
            transition_logits = transition_model(embedding)
            value_logits = value_model(embedding)
            return value_logits, policy_logits, transition_logits

        return init, (embed_model, value_model, policy_model, transition_model)

    return hk.multi_transform(f)
