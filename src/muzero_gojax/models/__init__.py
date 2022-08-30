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

    def f():
        # pylint: disable=invalid-name
        embed_model = {
            'identity': embed.Identity, 'linear_conv': embed.LinearConvEmbed,
            'black_perspective': embed.BlackPerspective, 'black_cnn_lite': embed.BlackCnnLite,
            'black_cnn_intermediate': embed.BlackCnnIntermediate, 'cnn_lite': embed.CnnLiteEmbed,
            'cnn_intermediate': embed.CnnIntermediateEmbed
        }[absl_flags.embed_model](absl_flags)
        value_model = {
            'random': value.RandomValue, 'linear': value.Linear3DValue,
            'linear_conv': value.LinearConvValue, 'cnn_lite': value.CnnLiteValue,
            'resnet_intermediate': value.ResnetIntermediateValue,
            'tromp_taylor': value.TrompTaylorValue
        }[absl_flags.value_model](absl_flags)
        policy_model = {
            'random': policy.RandomPolicy, 'linear': policy.Linear3DPolicy,
            'cnn_lite': policy.CnnLitePolicy,
            'resnet_intermediate': policy.ResnetIntermediatePolicy,
            'tromp_taylor': policy.TrompTaylorPolicy
        }[absl_flags.policy_model](absl_flags)
        transition_model = {
            'real': transition.RealTransition, 'black_perspective': transition.BlackRealTransition,
            'random': transition.RandomTransition, 'linear_conv': transition.LinearConvTransition,
            'cnn_lite': transition.CnnLiteTransition,
            'cnn_intermediate': transition.CnnIntermediateTransition,
            'resnet_intermediate': transition.ResnetIntermediateTransition,
        }[absl_flags.transition_model](absl_flags)

        def init(states):
            embedding = embed_model(states)
            policy_logits = policy_model(embedding)
            transition_logits = transition_model(embedding)
            value_logits = value_model(embedding)
            return value_logits, policy_logits, transition_logits

        return init, (embed_model, value_model, policy_model, transition_model)

    return hk.multi_transform(f)
