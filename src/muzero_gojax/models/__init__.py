"""High-level model management."""
# pylint:disable=duplicate-code
import haiku as hk

from muzero_gojax.models import decode
from muzero_gojax.models import embed
from muzero_gojax.models import policy
from muzero_gojax.models import transition
from muzero_gojax.models import value

EMBED_INDEX = 0
DECODE_INDEX = 1
VALUE_INDEX = 2
POLICY_INDEX = 3
TRANSITION_INDEX = 4


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
            'black_cnn_medium': embed.BlackCnnMedium, 'cnn_lite': embed.CnnLiteEmbed,
            'cnn_medium': embed.CnnMediumEmbed, 'resnet': embed.ResNetV2Embed,
        }[absl_flags.embed_model](absl_flags)
        decode_model = {
            'noop': decode.NoOpDecode, 'resnet': decode.ResNetV2Decode
        }[absl_flags.decode_model](absl_flags)
        value_model = {
            'random': value.RandomValue, 'linear': value.Linear3DValue,
            'linear_conv': value.LinearConvValue, 'cnn_lite': value.CnnLiteValue,
            'resnet_medium': value.ResnetMediumValue, 'tromp_taylor': value.TrompTaylorValue
        }[absl_flags.value_model](absl_flags)
        policy_model = {
            'random': policy.RandomPolicy, 'linear': policy.Linear3DPolicy,
            'linear_conv': policy.LinearConvPolicy, 'cnn_lite': policy.CnnLitePolicy,
            'resnet_medium': policy.ResnetMediumPolicy, 'tromp_taylor': policy.TrompTaylorPolicy
        }[absl_flags.policy_model](absl_flags)
        transition_model = {
            'real': transition.RealTransition, 'black_perspective': transition.BlackRealTransition,
            'random': transition.RandomTransition, 'linear_conv': transition.LinearConvTransition,
            'cnn_lite': transition.CnnLiteTransition, 'cnn_medium': transition.CnnMediumTransition,
            'resnet_medium': transition.ResnetMediumTransition,
            'resnet': transition.ResNetV2Transition,
        }[absl_flags.transition_model](absl_flags)

        def init(states):
            embedding = embed_model(states)
            policy_logits = policy_model(embedding)
            transition_logits = transition_model(embedding)
            value_logits = value_model(embedding)
            return value_logits, policy_logits, transition_logits

        return init, (embed_model, decode_model, value_model, policy_model, transition_model)

    return hk.multi_transform(f)
