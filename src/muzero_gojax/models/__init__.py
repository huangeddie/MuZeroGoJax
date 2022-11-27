"""High-level model management."""
# pylint:disable=duplicate-code

import os
import pickle
from types import ModuleType
from typing import Callable, Tuple

import chex
import gojax
import haiku as hk
import jax.numpy as jnp
import jax.random
import jax.tree_util
import optax
from absl import flags

from muzero_gojax import nt_utils
from muzero_gojax.models import base, decode, embed, policy, transition, value

_EMBED_MODEL = flags.DEFINE_string(
    'embed_model', 'LinearConvEmbed', 'Class name of the submodel to use. '
    'See the submodel module to view all submodel classes.')
_DECODE_MODEL = flags.DEFINE_string(
    'decode_model', 'LinearConvDecode', 'Class name of the submodel to use. '
    'See the submodel module to view all submodel classes.')
_VALUE_MODEL = flags.DEFINE_string(
    'value_model', 'LinearConvValue', 'Class name of the submodel to use. '
    'See the submodel module to view all submodel classes.')
_POLICY_MODEL = flags.DEFINE_string(
    'policy_model', 'LinearConvPolicy', 'Class name of the submodel to use. '
    'See the submodel module to view all submodel classes.')
_TRANSITION_MODEL = flags.DEFINE_string(
    'transition_model', 'LinearConvTransition',
    'Class name of the submodel to use. '
    'See the submodel module to view all submodel classes.')

_EMBED_DIM = flags.DEFINE_integer('embed_dim', 6, 'Embedded dimension size.')
_HDIM = flags.DEFINE_integer('hdim', 32, 'Hidden dimension size.')
_EMBED_NLAYERS = flags.DEFINE_integer('embed_nlayers', 0,
                                      'Number of embed layers.')
_VALUE_NLAYERS = flags.DEFINE_integer('value_nlayers', 0,
                                      'Number of value layers.')
_DECODE_NLAYERS = flags.DEFINE_integer('decode_nlayers', 0,
                                       'Number of decode layers.')
_POLICY_NLAYERS = flags.DEFINE_integer('policy_nlayers', 0,
                                       'Number of policy layers.')
_TRANSITION_NLAYERS = flags.DEFINE_integer('transition_nlayers', 0,
                                           'Number of transition layers.')

_LOAD_DIR = flags.DEFINE_string(
    'load_dir', None, 'File path to load the saved parameters.'
    'Otherwise the model starts from randomly '
    'initialized weights.')

EMBED_INDEX = 0
DECODE_INDEX = 1
VALUE_INDEX = 2
POLICY_INDEX = 3
TRANSITION_INDEX = 4


@chex.dataclass(frozen=True)
class PolicyOutput:
    """Policy output."""
    # N
    sampled_actions: jnp.ndarray
    # N x A'
    visited_actions: jnp.ndarray
    # N x A'
    visited_qvalues: jnp.ndarray


# RNG, Go State -> Action.
PolicyModel = Callable[[jax.random.KeyArray, jnp.ndarray], PolicyOutput]


def load_tree_array(filepath: str, dtype: str = None) -> dict:
    """Loads the parameters casted into an optional type"""
    with open(filepath, 'rb') as file_array:
        tree = pickle.load(file_array)
    if dtype:
        tree = jax.tree_util.tree_map(lambda x: x.astype(dtype), tree)
    return tree


def _fetch_submodel(
        submodel_module: ModuleType,
        submodel_build_config: base.SubModelBuildConfig,
        model_build_config: base.ModelBuildConfig) -> base.BaseGoModel:
    model_registry = dict([(name, cls)
                           for name, cls in submodel_module.__dict__.items()
                           if isinstance(cls, type)])
    return model_registry[submodel_build_config.name_key](
        model_build_config, submodel_build_config)


def _build_model_transform(
    model_build_config: base.ModelBuildConfig,
    embed_build_config: base.SubModelBuildConfig,
    decode_build_config: base.SubModelBuildConfig,
    value_build_config: base.SubModelBuildConfig,
    policy_build_config: base.SubModelBuildConfig,
    transition_build_config: base.SubModelBuildConfig,
) -> hk.MultiTransformed:
    """Builds a multi-transformed Go model."""

    def f():
        # pylint: disable=invalid-name
        embed_model = _fetch_submodel(embed, embed_build_config,
                                      model_build_config)
        decode_model = _fetch_submodel(decode, decode_build_config,
                                       model_build_config)
        value_model = _fetch_submodel(value, value_build_config,
                                      model_build_config)
        policy_model = _fetch_submodel(policy, policy_build_config,
                                       model_build_config)
        transition_model = _fetch_submodel(transition, transition_build_config,
                                           model_build_config)

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

    model_build_config = base.ModelBuildConfig(board_size=board_size,
                                               hdim=_HDIM.value,
                                               embed_dim=_EMBED_DIM.value,
                                               dtype=dtype)
    embed_build_config = base.SubModelBuildConfig(name_key=_EMBED_MODEL.value,
                                                  nlayers=_EMBED_NLAYERS.value)
    decode_build_config = base.SubModelBuildConfig(
        name_key=_DECODE_MODEL.value, nlayers=_DECODE_NLAYERS.value)
    value_build_config = base.SubModelBuildConfig(name_key=_VALUE_MODEL.value,
                                                  nlayers=_VALUE_NLAYERS.value)
    policy_build_config = base.SubModelBuildConfig(
        name_key=_POLICY_MODEL.value, nlayers=_POLICY_NLAYERS.value)
    transition_build_config = base.SubModelBuildConfig(
        name_key=_TRANSITION_MODEL.value, nlayers=_TRANSITION_NLAYERS.value)

    go_model = _build_model_transform(
        model_build_config,
        embed_build_config,
        decode_build_config,
        value_build_config,
        policy_build_config,
        transition_build_config,
    )
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
        base.ModelBuildConfig(embed_dim=gojax.NUM_CHANNELS),
        embed_build_config=base.SubModelBuildConfig(name_key='IdentityEmbed'),
        decode_build_config=base.SubModelBuildConfig(
            name_key='AmplifiedDecode'),
        value_build_config=base.SubModelBuildConfig(name_key='RandomValue'),
        policy_build_config=base.SubModelBuildConfig(name_key='RandomPolicy'),
        transition_build_config=base.SubModelBuildConfig(
            name_key='RandomTransition'),
    )


def make_random_policy_tromp_taylor_value_model():
    """Random normal policy with tromp taylor value."""
    return _build_model_transform(
        base.ModelBuildConfig(embed_dim=gojax.NUM_CHANNELS),
        embed_build_config=base.SubModelBuildConfig(name_key='IdentityEmbed'),
        decode_build_config=base.SubModelBuildConfig(
            name_key='AmplifiedDecode'),
        value_build_config=base.SubModelBuildConfig(
            name_key='TrompTaylorValue'),
        policy_build_config=base.SubModelBuildConfig(name_key='RandomPolicy'),
        transition_build_config=base.SubModelBuildConfig(
            name_key='RealTransition'),
    )


def make_tromp_taylor_model():
    """Makes a Tromp Taylor (greedy) model."""
    return _build_model_transform(
        base.ModelBuildConfig(embed_dim=gojax.NUM_CHANNELS),
        embed_build_config=base.SubModelBuildConfig(name_key='IdentityEmbed'),
        decode_build_config=base.SubModelBuildConfig(
            name_key='AmplifiedDecode'),
        value_build_config=base.SubModelBuildConfig(
            name_key='TrompTaylorValue'),
        policy_build_config=base.SubModelBuildConfig(
            name_key='TrompTaylorPolicy'),
        transition_build_config=base.SubModelBuildConfig(
            name_key='RealTransition'))


def make_tromp_taylor_amplified_model():
    """Makes a Tromp Taylor amplified (greedy) model."""
    return _build_model_transform(
        base.ModelBuildConfig(embed_dim=gojax.NUM_CHANNELS),
        embed_build_config=base.SubModelBuildConfig(name_key='IdentityEmbed'),
        decode_build_config=base.SubModelBuildConfig(
            name_key='AmplifiedDecode'),
        value_build_config=base.SubModelBuildConfig(
            name_key='TrompTaylorValue'),
        policy_build_config=base.SubModelBuildConfig(
            name_key='TrompTaylorAmplifiedPolicy'),
        transition_build_config=base.SubModelBuildConfig(
            name_key='RealTransition'))


def get_policy_model(go_model: hk.MultiTransformed,
                     params: optax.Params,
                     sample_action_size: int = 0) -> PolicyModel:
    """Returns policy model function of the go model.

    Args:
        go_model (hk.MultiTransformed): Go model.
        params (optax.Params): Parameters.
        sample_action_size (int): Sample action size at each tree level.
            `m` in the Gumbel MuZero paper.
    Returns:
        jax.tree_util.Partial: Policy model.
    """

    if sample_action_size <= 0:

        def policy_fn(rng_key: jax.random.KeyArray, states: jnp.ndarray):
            embeds = go_model.apply[EMBED_INDEX](params, rng_key, states)
            policy_logits = go_model.apply[POLICY_INDEX](params, rng_key,
                                                         embeds)
            gumbel = jax.random.gumbel(rng_key,
                                       shape=policy_logits.shape,
                                       dtype=policy_logits.dtype)
            return PolicyOutput(sampled_actions=jnp.argmax(
                policy_logits + gumbel, axis=-1).astype('uint16'),
                                visited_actions=None,
                                visited_qvalues=None)
    else:

        def policy_fn(rng_key: jax.random.KeyArray, states: jnp.ndarray):
            embeds = go_model.apply[EMBED_INDEX](params, rng_key, states)
            batch_size, hdim, board_size, _ = embeds.shape
            policy_logits = go_model.apply[POLICY_INDEX](params, rng_key,
                                                         embeds)
            gumbel = jax.random.gumbel(rng_key,
                                       shape=policy_logits.shape,
                                       dtype=policy_logits.dtype)
            _, sampled_actions = jax.lax.top_k(policy_logits + gumbel,
                                               k=sample_action_size)
            chex.assert_shape(sampled_actions,
                              (batch_size, sample_action_size))
            # N x A' x D x B x B
            partial_transitions = go_model.apply[TRANSITION_INDEX](
                params, rng_key, embeds, batch_partial_actions=sampled_actions)
            chex.assert_shape(
                partial_transitions,
                (batch_size, sample_action_size, hdim, board_size, board_size))
            partial_transition_value_logits = nt_utils.unflatten_first_dim(
                go_model.apply[VALUE_INDEX](
                    params, rng_key,
                    nt_utils.flatten_first_two_dims(partial_transitions)),
                batch_size, sample_action_size)
            chex.assert_shape(partial_transition_value_logits,
                              (batch_size, sample_action_size))
            # We take the negative of the transition logits because they're in
            # the opponent's perspective.
            argmax_of_top_m = jnp.argmax(-partial_transition_value_logits,
                                         axis=1)
            return PolicyOutput(sampled_actions=sampled_actions[
                jnp.arange(len(sampled_actions)),
                argmax_of_top_m].astype('uint16'),
                                visited_actions=None,
                                visited_qvalues=None)

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
