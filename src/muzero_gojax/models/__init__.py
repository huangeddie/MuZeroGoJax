"""High-level model management."""
import os
import pickle
from typing import Tuple

# pylint:disable=duplicate-code
import gojax
import haiku as hk
import jax.random
import jax.tree_util
import optax
from absl import flags

from muzero_gojax.models import base
from muzero_gojax.models import decode
from muzero_gojax.models import embed
from muzero_gojax.models import policy
from muzero_gojax.models import transition
from muzero_gojax.models import value

_EMBED_MODEL = flags.DEFINE_enum('embed_model', 'linear_conv',
                                 ['black_perspective', 'identity', 'amplified', 'linear_conv',
                                  'cnn_lite', 'black_cnn_lite', 'resnet'],
                                 'State embedding model architecture.')
_DECODE_MODEL = flags.DEFINE_enum('decode_model', 'linear_conv',
                                  ['amplified', 'resnet', 'linear_conv'],
                                  'State decoding model architecture.')
_VALUE_MODEL = flags.DEFINE_enum('value_model', 'linear_conv',
                                 ['random', 'linear', 'linear_conv', 'cnn_lite', 'resnet',
                                  'tromp_taylor'], 'Value model architecture.')
_POLICY_MODEL = flags.DEFINE_enum('policy_model', 'linear_conv',
                                  ['random', 'linear', 'linear_conv', 'cnn_lite', 'resnet',
                                   'tromp_taylor'], 'Policy model architecture.')
_TRANSITION_MODEL = flags.DEFINE_enum('transition_model', 'linear_conv',
                                      ['real', 'black_perspective', 'random', 'linear_conv',
                                       'cnn_lite', 'resnet', 'resnet_action_embed'],
                                      'Transition model architecture.')

_HDIM = flags.DEFINE_integer('hdim', 32, 'Hidden dimension size.')
_NLAYERS = flags.DEFINE_integer('nlayers', 1, 'Number of layers. Applicable to ResNetV2 models.')
_EMBED_DIM = flags.DEFINE_integer('embed_dim', 6, 'Embedded dimension size.')

_LOAD_DIR = flags.DEFINE_string('load_dir', None, 'File path to load the saved parameters.'
                                                  'Otherwise the model starts from randomly '
                                                  'initialized weights.')

EMBED_INDEX = 0
DECODE_INDEX = 1
VALUE_INDEX = 2
POLICY_INDEX = 3
TRANSITION_INDEX = 4


def load_tree_array(filepath: str, dtype: str = None) -> dict:
    """Loads the parameters casted into an optional type"""
    with open(filepath, 'rb') as file_array:
        tree = pickle.load(file_array)
    if dtype:
        tree = jax.tree_util.tree_map(lambda x: x.astype(dtype), tree)
    return tree


def build_model(board_size: int) -> Tuple[hk.MultiTransformed, optax.Params]:
    """
    Builds the corresponding model for the given name.

    :param board_size: Board size
    :return: A Haiku multi-transformed Go model consisting of (1) a state embedding model,
    (2) a policy model, (3) a transition model, and (4) a value model.
    """

    model_build_params = base.ModelBuildParams(board_size, _HDIM.value, _NLAYERS.value,
                                               _EMBED_DIM.value, _EMBED_MODEL.value,
                                               _DECODE_MODEL.value, _VALUE_MODEL.value,
                                               _POLICY_MODEL.value, _TRANSITION_MODEL.value)

    go_model = build_model_transform(model_build_params)
    if _LOAD_DIR.value:
        params = load_tree_array(os.path.join(_LOAD_DIR.value, 'params.npz'), dtype='bfloat16')
        print(f"Loaded parameters from '{_LOAD_DIR.value}'.")
    else:
        params = go_model.init(jax.random.PRNGKey(42), gojax.new_states(board_size, 1))
        print("Initialized parameters randomly.")
    return go_model, params


def build_model_transform(model_build_params: base.ModelBuildParams) -> hk.MultiTransformed:
    """Builds a multi-transformed Go model."""

    def f():
        # pylint: disable=invalid-name
        embed_model = {
            'identity': embed.IdentityEmbed, 'linear_conv': embed.LinearConvEmbed,
            'amplified': embed.AmplifiedEmbed, 'black_perspective': embed.BlackPerspectiveEmbed,
            'black_cnn_lite': embed.BlackCnnLiteEmbed, 'cnn_lite': embed.CnnLiteEmbed,
            'resnet': embed.ResNetV2Embed,
        }[model_build_params.embed_model_key](model_build_params)
        decode_model = {
            'amplified': decode.AmplifiedDecode, 'resnet': decode.ResNetV2Decode,
            'linear_conv': decode.LinearConvDecode
        }[model_build_params.decode_model_key](model_build_params)
        value_model = {
            'random': value.RandomValue, 'linear': value.Linear3DValue,
            'linear_conv': value.LinearConvValue, 'cnn_lite': value.CnnLiteValue,
            'resnet': value.ResNetV2Value, 'tromp_taylor': value.TrompTaylorValue
        }[model_build_params.value_model_key](model_build_params)
        policy_model = {
            'random': policy.RandomPolicy, 'linear': policy.Linear3DPolicy,
            'linear_conv': policy.LinearConvPolicy, 'cnn_lite': policy.CnnLitePolicy,
            'resnet': policy.ResNetV2Policy, 'tromp_taylor': policy.TrompTaylorPolicy
        }[model_build_params.policy_model_key](model_build_params)
        transition_model = {
            'real': transition.RealTransition, 'black_perspective': transition.BlackRealTransition,
            'random': transition.RandomTransition, 'linear_conv': transition.LinearConvTransition,
            'cnn_lite': transition.CnnLiteTransition, 'resnet': transition.ResNetV2Transition,
            'resnet_action_embed': transition.ResNetV2ActionEmbedTransition
        }[model_build_params.transition_model_key](model_build_params)

        def init(states):
            embedding = embed_model(states)
            decoding = decode_model(embedding)
            policy_logits = policy_model(embedding)
            transition_logits = transition_model(embedding)
            value_logits = value_model(embedding)
            return decoding, value_logits, policy_logits, transition_logits

        return init, (embed_model, decode_model, value_model, policy_model, transition_model)

    return hk.multi_transform(f)


def save_model(params: optax.Params, model_dir: str):
    """
    Saves the parameters with a filename that is the hash of the flags.

    :param params: Model parameters.
    :param model_dir: Sub model directory to dump all data in.
    :return: None or the model directory.
    """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    params_filename = os.path.join(model_dir, 'params.npz')
    with open(params_filename, 'wb') as params_file:
        pickle.dump(jax.tree_util.tree_map(lambda x: x.astype('float32'), params), params_file)
    print(f"Saved model to '{model_dir}'.")
