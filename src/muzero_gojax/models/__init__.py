"""High-level model management."""
# pylint:disable=duplicate-code
import haiku as hk
from absl import flags

from muzero_gojax.models import decode
from muzero_gojax.models import embed
from muzero_gojax.models import policy
from muzero_gojax.models import transition
from muzero_gojax.models import value

_EMBED_MODEL = flags.DEFINE_enum('embed_model', 'black_perspective',
                                 ['black_perspective', 'identity', 'linear_conv', 'cnn_lite',
                                  'black_cnn_lite', 'resnet'],
                                 'State embedding model architecture.')
_DECODE_MODEL = flags.DEFINE_enum('decode_model', 'noop', ['noop', 'resnet', 'linear_conv'],
                                  'State decoding model architecture.')
_VALUE_MODEL = flags.DEFINE_enum('value_model', 'linear',
                                 ['random', 'linear', 'linear_conv', 'cnn_lite', 'resnet_medium',
                                  'tromp_taylor'], 'Value model architecture.')
_POLICY_MODEL = flags.DEFINE_enum('policy_model', 'linear',
                                  ['random', 'linear', 'linear_conv', 'cnn_lite', 'resnet_medium',
                                   'tromp_taylor'], 'Policy model architecture.')
_TRANSITION_MODEL = flags.DEFINE_enum('transition_model', 'black_perspective',
                                      ['real', 'black_perspective', 'random', 'linear_conv',
                                       'cnn_lite', 'resnet_medium', 'resnet'],
                                      'Transition model architecture.')

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
            'cnn_lite': embed.CnnLiteEmbed, 'resnet': embed.ResNetV2Embed,
        }[_EMBED_MODEL.value](absl_flags)
        decode_model = {
            'noop': decode.NoOpDecode, 'resnet': decode.ResNetV2Decode,
            'linear_conv': decode.LinearConvDecode
        }[_DECODE_MODEL.value](absl_flags)
        value_model = {
            'random': value.RandomValue, 'linear': value.Linear3DValue,
            'linear_conv': value.LinearConvValue, 'cnn_lite': value.CnnLiteValue,
            'resnet_medium': value.ResnetMediumValue, 'tromp_taylor': value.TrompTaylorValue
        }[_VALUE_MODEL.value](absl_flags)
        policy_model = {
            'random': policy.RandomPolicy, 'linear': policy.Linear3DPolicy,
            'linear_conv': policy.LinearConvPolicy, 'cnn_lite': policy.CnnLitePolicy,
            'resnet_medium': policy.ResnetMediumPolicy, 'tromp_taylor': policy.TrompTaylorPolicy
        }[_POLICY_MODEL.value](absl_flags)
        transition_model = {
            'real': transition.RealTransition, 'black_perspective': transition.BlackRealTransition,
            'random': transition.RandomTransition, 'linear_conv': transition.LinearConvTransition,
            'cnn_lite': transition.CnnLiteTransition,
            'resnet_medium': transition.ResnetMediumTransition,
            'resnet': transition.ResNetV2Transition,
        }[_TRANSITION_MODEL.value](absl_flags)

        def init(states):
            embedding = embed_model(states)
            decoding = decode_model(embedding)
            policy_logits = policy_model(embedding)
            transition_logits = transition_model(embedding)
            value_logits = value_model(embedding)
            return decoding, value_logits, policy_logits, transition_logits

        return init, (embed_model, decode_model, value_model, policy_model, transition_model)

    return hk.multi_transform(f)
