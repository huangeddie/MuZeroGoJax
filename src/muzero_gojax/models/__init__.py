"""High-level model management."""
import os
import pickle
from typing import Callable, Tuple

# pylint:disable=duplicate-code
import gojax
import haiku as hk
import jax.numpy as jnp
import jax.random
import jax.tree_util
import optax
from absl import flags

from muzero_gojax.models import base, decode, embed, policy, transition, value

_EMBED_MODEL = flags.DEFINE_enum('embed_model', 'non_spatial_conv', [
    'black_perspective', 'identity', 'amplified', 'non_spatial_conv',
    'cnn_lite', 'black_cnn_lite', 'resnet'
], 'State embedding model architecture.')
_DECODE_MODEL = flags.DEFINE_enum(
    'decode_model', 'non_spatial_conv',
    ['amplified', 'scale', 'resnet', 'non_spatial_conv'],
    'State decoding model architecture.')
_VALUE_MODEL = flags.DEFINE_enum('value_model', 'non_spatial_conv', [
    'random',
    'linear',
    'non_spatial_conv',
    'non_spatial_quad_conv',
    'heuristic_quad_conv',
    'cnn_lite',
    'resnet',
    'tromp_taylor',
    'piece_counter',
], 'Value model architecture.')
_POLICY_MODEL = flags.DEFINE_enum('policy_model', 'non_spatial_conv', [
    'random', 'linear', 'non_spatial_conv', 'cnn_lite', 'resnet',
    'tromp_taylor'
], 'Policy model architecture.')
_TRANSITION_MODEL = flags.DEFINE_enum('transition_model', 'non_spatial_conv', [
    'real', 'black_perspective', 'random', 'non_spatial_conv', 'resnet_action'
], 'Transition model architecture.')

_HDIM = flags.DEFINE_integer('hdim', 32, 'Hidden dimension size.')
_NLAYERS = flags.DEFINE_integer(
    'nlayers', 0, 'Number of layers. Applicable to ResNetV2 models.')
_EMBED_DIM = flags.DEFINE_integer('embed_dim', 6, 'Embedded dimension size.')

_LOAD_DIR = flags.DEFINE_string(
    'load_dir', None, 'File path to load the saved parameters.'
    'Otherwise the model starts from randomly '
    'initialized weights.')

EMBED_INDEX = 0
DECODE_INDEX = 1
VALUE_INDEX = 2
POLICY_INDEX = 3
TRANSITION_INDEX = 4

PolicyModel = Callable[[jax.random.KeyArray, jnp.ndarray], jnp.ndarray]


def load_tree_array(filepath: str, dtype: str = None) -> dict:
    """Loads the parameters casted into an optional type"""
    with open(filepath, 'rb') as file_array:
        tree = pickle.load(file_array)
    if dtype:
        tree = jax.tree_util.tree_map(lambda x: x.astype(dtype), tree)
    return tree


def _build_model_transform(
        model_build_params: base.ModelBuildParams) -> hk.MultiTransformed:
    """Builds a multi-transformed Go model."""

    def f():
        # pylint: disable=invalid-name
        embed_model = {
            'identity': embed.IdentityEmbed,
            'non_spatial_conv': embed.NonSpatialConvEmbed,
            'amplified': embed.AmplifiedEmbed,
            'black_perspective': embed.BlackPerspectiveEmbed,
            'resnet': embed.ResNetV2Embed,
        }[model_build_params.embed_model_key](model_build_params)
        decode_model = {
            'amplified': decode.AmplifiedDecode,
            'scale': decode.ScaleDecode,
            'resnet': decode.ResNetV2Decode,
            'non_spatial_conv': decode.NonSpatialConvDecode
        }[model_build_params.decode_model_key](model_build_params)
        value_model = {
            'random': value.RandomValue,
            'linear': value.Linear3DValue,
            'non_spatial_conv': value.NonSpatialConvValue,
            'non_spatial_quad_conv': value.NonSpatialQuadConvValue,
            'heuristic_quad_conv': value.HeuristicQuadConvValue,
            'resnet': value.ResNetV2Value,
            'tromp_taylor': value.TrompTaylorValue,
            'piece_counter': value.PieceCounterValue,
        }[model_build_params.value_model_key](model_build_params)
        policy_model = {
            'random': policy.RandomPolicy,
            'linear': policy.Linear3DPolicy,
            'non_spatial_conv': policy.NonSpatialConvPolicy,
            'resnet': policy.ResNetV2Policy,
            'tromp_taylor': policy.TrompTaylorPolicy
        }[model_build_params.policy_model_key](model_build_params)
        transition_model = {
            'real': transition.RealTransition,
            'black_perspective': transition.BlackRealTransition,
            'random': transition.RandomTransition,
            'non_spatial_conv': transition.NonSpatialConvTransition,
            'resnet_action': transition.ResNetV2ActionTransition
        }[model_build_params.transition_model_key](model_build_params)

        def init(states):
            embedding = embed_model(states)
            decoding = decode_model(embedding)
            policy_logits = policy_model(embedding)
            transition_logits = transition_model(embedding)
            value_logits = value_model(embedding)
            return decoding, value_logits, policy_logits, transition_logits

        return init, (embed_model, decode_model, value_model, policy_model,
                      transition_model)

    return hk.multi_transform(f)


def build_model_with_params(
        board_size: int, dtype: str, rng_key: jax.random.KeyArray
) -> Tuple[hk.MultiTransformed, optax.Params]:
    """
    Builds the corresponding model for the given name.

    :param board_size: Board size
    :return: A Haiku multi-transformed Go model consisting of (1) a state embedding model,
    (2) a policy model, (3) a transition model, and (4) a value model.
    """

    model_build_params = base.ModelBuildParams(
        board_size, _HDIM.value, _NLAYERS.value, _EMBED_DIM.value, dtype,
        _EMBED_MODEL.value, _DECODE_MODEL.value, _VALUE_MODEL.value,
        _POLICY_MODEL.value, _TRANSITION_MODEL.value)

    go_model = _build_model_transform(model_build_params)
    if _LOAD_DIR.value:
        params = load_tree_array(os.path.join(_LOAD_DIR.value, 'params.npz'),
                                 dtype=dtype)
        print(f"Loaded parameters from '{_LOAD_DIR.value}'.")
    else:
        params = go_model.init(rng_key, gojax.new_states(board_size, 1))
        print("Initialized parameters randomly.")
    return go_model, params


def make_random_model():
    """Makes a random normal model."""
    return _build_model_transform(
        base.ModelBuildParams(embed_dim=gojax.NUM_CHANNELS,
                              embed_model_key='identity',
                              decode_model_key='amplified',
                              value_model_key='random',
                              policy_model_key='random',
                              transition_model_key='random'))


def get_policy_model(go_model: hk.MultiTransformed,
                     params: optax.Params) -> PolicyModel:
    """Returns policy model function of the go model.

    Args:
        go_model (hk.MultiTransformed): Go model.
        params (optax.Params): Parameters.

    Returns:
        jax.tree_util.Partial: Policy model.
    """

    def policy_fn(rng_key: jax.random.KeyArray, states: jnp.ndarray):
        embeds = go_model.apply[EMBED_INDEX](params, rng_key, states)
        return go_model.apply[POLICY_INDEX](params, rng_key, embeds)

    return policy_fn


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
        pickle.dump(
            jax.tree_util.tree_map(lambda x: x.astype('float32'), params),
            params_file)
    print(f"Saved model to '{model_dir}'.")
