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
from muzero_gojax.models import (_base, _build_config, _decode, _embed,
                                 _policy, _transition, _value)
# pylint: disable=unused-import
from muzero_gojax.models._build_config import *
from muzero_gojax.models._decode import *
from muzero_gojax.models._embed import *
from muzero_gojax.models._policy import *
from muzero_gojax.models._transition import *
from muzero_gojax.models._value import *

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
        submodel_build_config: _build_config.SubModelBuildConfig,
        model_build_config: _build_config.ModelBuildConfig
) -> _base.BaseGoModel:
    model_registry = dict([(name, cls)
                           for name, cls in submodel_module.__dict__.items()
                           if isinstance(cls, type)])
    return model_registry[submodel_build_config.name_key](
        model_build_config, submodel_build_config)


def _build_model_transform(
    all_model_build_configs: _build_config.AllModelBuildConfigs
) -> hk.MultiTransformed:
    """Builds a multi-transformed Go model."""

    def f():
        # pylint: disable=invalid-name
        embed_model = _fetch_submodel(
            _embed, all_model_build_configs.embed_build_config,
            all_model_build_configs.model_build_config)
        decode_model = _fetch_submodel(
            _decode, all_model_build_configs.decode_build_config,
            all_model_build_configs.model_build_config)
        value_model = _fetch_submodel(
            _value, all_model_build_configs.value_build_config,
            all_model_build_configs.model_build_config)
        policy_model = _fetch_submodel(
            _policy, all_model_build_configs.policy_build_config,
            all_model_build_configs.model_build_config)
        transition_model = _fetch_submodel(
            _transition, all_model_build_configs.transition_build_config,
            all_model_build_configs.model_build_config)

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

    all_model_build_configs = get_all_model_build_configs(board_size, dtype)

    go_model = _build_model_transform(all_model_build_configs)
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
    all_model_build_configs = _build_config.AllModelBuildConfigs(
        model_build_config=_build_config.ModelBuildConfig(
            embed_dim=gojax.NUM_CHANNELS),
        embed_build_config=_build_config.SubModelBuildConfig(
            name_key='IdentityEmbed'),
        decode_build_config=_build_config.SubModelBuildConfig(
            name_key='AmplifiedDecode'),
        value_build_config=_build_config.SubModelBuildConfig(
            name_key='RandomValue'),
        policy_build_config=_build_config.SubModelBuildConfig(
            name_key='RandomPolicy'),
        transition_build_config=_build_config.SubModelBuildConfig(
            name_key='RandomTransition'),
    )
    return _build_model_transform(all_model_build_configs)


def make_random_policy_tromp_taylor_value_model():
    """Random normal policy with tromp taylor value."""
    all_model_build_configs = _build_config.AllModelBuildConfigs(
        model_build_config=_build_config.ModelBuildConfig(
            embed_dim=gojax.NUM_CHANNELS),
        embed_build_config=_build_config.SubModelBuildConfig(
            name_key='IdentityEmbed'),
        decode_build_config=_build_config.SubModelBuildConfig(
            name_key='AmplifiedDecode'),
        value_build_config=_build_config.SubModelBuildConfig(
            name_key='TrompTaylorValue'),
        policy_build_config=_build_config.SubModelBuildConfig(
            name_key='RandomPolicy'),
        transition_build_config=_build_config.SubModelBuildConfig(
            name_key='RealTransition'),
    )
    return _build_model_transform(all_model_build_configs)


def make_tromp_taylor_model():
    """Makes a Tromp Taylor (greedy) model."""
    all_model_build_configs = _build_config.AllModelBuildConfigs(
        model_build_config=_build_config.ModelBuildConfig(
            embed_dim=gojax.NUM_CHANNELS),
        embed_build_config=_build_config.SubModelBuildConfig(
            name_key='IdentityEmbed'),
        decode_build_config=_build_config.SubModelBuildConfig(
            name_key='AmplifiedDecode'),
        value_build_config=_build_config.SubModelBuildConfig(
            name_key='TrompTaylorValue'),
        policy_build_config=_build_config.SubModelBuildConfig(
            name_key='TrompTaylorPolicy'),
        transition_build_config=_build_config.SubModelBuildConfig(
            name_key='RealTransition'))
    return _build_model_transform(all_model_build_configs)


def make_tromp_taylor_amplified_model():
    """Makes a Tromp Taylor amplified (greedy) model."""
    all_model_build_configs = _build_config.AllModelBuildConfigs(
        model_build_config=_build_config.ModelBuildConfig(
            embed_dim=gojax.NUM_CHANNELS),
        embed_build_config=_build_config.SubModelBuildConfig(
            name_key='IdentityEmbed'),
        decode_build_config=_build_config.SubModelBuildConfig(
            name_key='AmplifiedDecode'),
        value_build_config=_build_config.SubModelBuildConfig(
            name_key='TrompTaylorValue'),
        policy_build_config=_build_config.SubModelBuildConfig(
            name_key='TrompTaylorAmplifiedPolicy'),
        transition_build_config=_build_config.SubModelBuildConfig(
            name_key='RealTransition'))
    return _build_model_transform(all_model_build_configs)


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


def hash_model_flags(board_size: int, dtype: str) -> int:
    """Hashes all model config related flags."""
    return hash(_build_config.get_all_model_build_configs(board_size, dtype))


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
