"""High-level model management."""
# pylint:disable=duplicate-code,unused-import

import dataclasses
import json
import os
import pickle
from types import ModuleType
from typing import Callable, List, Tuple

import chex
import haiku as hk
import jax.numpy as jnp
import jax.random
import jax.tree_util
import jmp
import optax
from absl import flags

import gojax
from muzero_gojax import drive, logger, nt_utils
from muzero_gojax.models import (_area, _base, _build_config, _embed, _policy,
                                 _transition, _value)
from muzero_gojax.models._area import *
from muzero_gojax.models._build_config import *
from muzero_gojax.models._embed import *
from muzero_gojax.models._policy import *
from muzero_gojax.models._transition import *
from muzero_gojax.models._value import *

_MIXED_PRECISION_POLICY = flags.DEFINE_string(
    'mixed_precision_policy',
    'params=float32,compute=bfloat16,output=bfloat16',
    'Mixed precision policy.')
_NORM_MIXED_PRECISION_POLICY = flags.DEFINE_string(
    'norm_mixed_precision_policy',
    'params=float32,compute=float32,output=bfloat16',
    'Mixed precision policy for normalization layers.')
_TRAINED_MODELS_DIR = flags.DEFINE_string(
    'trained_models_dir', './trained_models/',
    'Directory containing trained weights.')
_QVAL_SCALE = flags.DEFINE_float(
    'qval_scale', 1.0, 'Q-value scale. '
    'Sigma from the MuZero Go paper.')

EMBED_INDEX = 0
AREA_INDEX = 1
VALUE_INDEX = 2
POLICY_INDEX = 3
TRANSITION_INDEX = 4


@chex.dataclass(frozen=True)
class PolicyOutput:
    """Policy output."""
    # N
    sampled_actions: jnp.ndarray
    # TODO: Fill out and use the fields below.
    # N x A'
    visited_actions: jnp.ndarray
    # N x A'
    visited_qvalues: jnp.ndarray


@chex.dataclass(frozen=True)
class ValueOutput:
    """Value output."""
    # N
    value: jnp.ndarray


# RNG, Go State -> Action.
PolicyModel = Callable[[jax.random.KeyArray, jnp.ndarray], PolicyOutput]

# RNG, Go State -> Value.
ValueModel = Callable[[jax.random.KeyArray, jnp.ndarray], ValueOutput]


@chex.dataclass(frozen=True)
class Benchmark:
    """Benchmark model."""
    policy: PolicyModel
    name: str


def _fetch_submodel(
        submodel_module: ModuleType,
        submeta_build_config: _build_config.ComponentBuildConfig,
        meta_build_config: _build_config.MetaBuildConfig) -> _base.BaseGoModel:
    model_registry = dict([(name, cls)
                           for name, cls in submodel_module.__dict__.items()
                           if isinstance(cls, type)])
    return model_registry[submeta_build_config.name_key](meta_build_config,
                                                         submeta_build_config)


def _set_mixed_policies():
    for submodel_module in [_embed, _area, _value, _policy, _transition]:
        for model_class in submodel_module.__dict__.values():
            if not isinstance(model_class, type):
                continue
            hk.mixed_precision.set_policy(
                model_class, jmp.get_policy(_MIXED_PRECISION_POLICY.value))


def _build_model_transform(
        model_build_config: _build_config.ModelBuildConfig
) -> hk.MultiTransformed:
    """Builds a multi-transformed Go model."""
    hk.mixed_precision.set_policy(
        hk.LayerNorm, jmp.get_policy(_NORM_MIXED_PRECISION_POLICY.value))
    _set_mixed_policies()

    def f():
        # pylint: disable=invalid-name
        embed_model = _fetch_submodel(_embed,
                                      model_build_config.embed_build_config,
                                      model_build_config.meta_build_config)
        area_model = _fetch_submodel(_area,
                                     model_build_config.area_build_config,
                                     model_build_config.meta_build_config)
        value_model = _fetch_submodel(_value,
                                      model_build_config.value_build_config,
                                      model_build_config.meta_build_config)
        policy_model = _fetch_submodel(_policy,
                                       model_build_config.policy_build_config,
                                       model_build_config.meta_build_config)
        transition_model = _fetch_submodel(
            _transition, model_build_config.transition_build_config,
            model_build_config.meta_build_config)

        def init(states):
            embedding = embed_model(states)
            decoding = area_model(embedding)
            policy_logits = policy_model(embedding)
            transition_logits = transition_model(embedding)
            value_logits = value_model(embedding)
            return decoding, value_logits, policy_logits, transition_logits

        return init, (embed_model, area_model, value_model, policy_model,
                      transition_model)

    return hk.multi_transform(f)


def build_model_with_params(
        model_build_config: _build_config.ModelBuildConfig,
        rng_key: jax.random.KeyArray
) -> Tuple[hk.MultiTransformed, optax.Params]:
    """
    Builds the corresponding model for the given name.

    :param board_size: Board size
    :return: A Haiku multi-transformed Go model consisting of (1) a state embedding model,
    (2) a policy model, (3) a transition model, and (4) a value model.
    """

    go_model = _build_model_transform(model_build_config)
    params = go_model.init(
        rng_key,
        gojax.new_states(model_build_config.meta_build_config.board_size, 1))
    logger.log("Initialized parameters randomly.")
    return go_model, params


def make_random_model():
    """Makes a random normal model."""
    model_build_config = _build_config.ModelBuildConfig(
        meta_build_config=_build_config.MetaBuildConfig(
            embed_dim=gojax.NUM_CHANNELS),
        embed_build_config=_build_config.ComponentBuildConfig(
            name_key='IdentityEmbed'),
        area_build_config=_build_config.ComponentBuildConfig(
            name_key='RandomArea'),
        value_build_config=_build_config.ComponentBuildConfig(
            name_key='RandomValue'),
        policy_build_config=_build_config.ComponentBuildConfig(
            name_key='RandomPolicy'),
        transition_build_config=_build_config.ComponentBuildConfig(
            name_key='RandomTransition'),
    )
    return _build_model_transform(model_build_config)


def make_random_policy_tromp_taylor_value_model():
    """Random normal policy with tromp taylor value."""
    model_build_config = _build_config.ModelBuildConfig(
        meta_build_config=_build_config.MetaBuildConfig(
            embed_dim=gojax.NUM_CHANNELS),
        embed_build_config=_build_config.ComponentBuildConfig(
            name_key='IdentityEmbed'),
        area_build_config=_build_config.ComponentBuildConfig(
            name_key='RandomArea'),
        value_build_config=_build_config.ComponentBuildConfig(
            name_key='TrompTaylorValue'),
        policy_build_config=_build_config.ComponentBuildConfig(
            name_key='RandomPolicy'),
        transition_build_config=_build_config.ComponentBuildConfig(
            name_key='RealTransition'),
    )
    return _build_model_transform(model_build_config)


def make_tromp_taylor_model():
    """Makes a Tromp Taylor (greedy) model."""
    model_build_config = _build_config.ModelBuildConfig(
        meta_build_config=_build_config.MetaBuildConfig(
            embed_dim=gojax.NUM_CHANNELS),
        embed_build_config=_build_config.ComponentBuildConfig(
            name_key='IdentityEmbed'),
        area_build_config=_build_config.ComponentBuildConfig(
            name_key='RandomArea'),
        value_build_config=_build_config.ComponentBuildConfig(
            name_key='TrompTaylorValue'),
        policy_build_config=_build_config.ComponentBuildConfig(
            name_key='TrompTaylorPolicy'),
        transition_build_config=_build_config.ComponentBuildConfig(
            name_key='RealTransition'))
    return _build_model_transform(model_build_config)


def make_tromp_taylor_amplified_model():
    """Makes a Tromp Taylor amplified (greedy) model."""
    model_build_config = _build_config.ModelBuildConfig(
        meta_build_config=_build_config.MetaBuildConfig(
            embed_dim=gojax.NUM_CHANNELS),
        embed_build_config=_build_config.ComponentBuildConfig(
            name_key='IdentityEmbed'),
        area_build_config=_build_config.ComponentBuildConfig(
            name_key='RandomArea'),
        value_build_config=_build_config.ComponentBuildConfig(
            name_key='TrompTaylorValue'),
        policy_build_config=_build_config.ComponentBuildConfig(
            name_key='TrompTaylorAmplifiedPolicy'),
        transition_build_config=_build_config.ComponentBuildConfig(
            name_key='RealTransition'))
    return _build_model_transform(model_build_config)


def get_benchmarks(board_size: int) -> List[Benchmark]:
    """Returns the set of all benchmarks compatible with the board size.

    Includes trained models.
    """
    benchmarks: List[Benchmark] = [
        Benchmark(policy=get_policy_model(make_tromp_taylor_amplified_model(),
                                          params={}),
                  name='Tromp Taylor Amplified')
    ]

    if os.path.exists(_TRAINED_MODELS_DIR.value):
        for item in os.listdir(_TRAINED_MODELS_DIR.value):
            model_dir = os.path.join(
                _TRAINED_MODELS_DIR.value,
                item,
            )
            if os.path.isdir(model_dir):
                with open(os.path.join(model_dir, 'build_config.json'),
                          'rt',
                          encoding='utf-8') as config_fp:
                    json_dict = json.load(config_fp)
                    meta_build_config = _build_config.MetaBuildConfig(
                        **json_dict['meta_build_config'])
                    if meta_build_config.board_size != board_size:
                        continue
                try:
                    go_model, trained_params, all_models_config = load_model(
                        model_dir)
                    if all_models_config.meta_build_config.board_size != board_size:
                        continue
                    base_trained_policy = get_policy_model(
                        go_model, trained_params)
                    benchmarks.append(
                        Benchmark(policy=base_trained_policy, name=model_dir))
                except OSError as os_error:
                    logger.log(
                        f"Failed to load model from {model_dir}: {os_error}")

    return benchmarks


def scale_q_complete(q_complete: jnp.ndarray):
    """Scales the q_complete value. This is sigma from the MuZero Go paper."""
    return _QVAL_SCALE.value * q_complete


def get_tromp_taylor_score(final_area_logits: jnp.ndarray) -> jnp.ndarray:
    """Difference between sigmoid sum of the player's area and opponent's area."""
    chex.assert_rank(final_area_logits, 4)
    final_areas = jax.nn.sigmoid(final_area_logits)
    return jnp.sum(final_areas[:, 0], axis=(1, 2)) - jnp.sum(final_areas[:, 1],
                                                             axis=(1, 2))


def get_value_model(go_model: hk.MultiTransformed,
                    params: optax.Params) -> ValueModel:
    """Returns value model function of the go model."""

    def value_fn(rng_key: jax.random.KeyArray,
                 states: jnp.ndarray) -> ValueOutput:
        embeds = go_model.apply[EMBED_INDEX](params, rng_key, states)
        value_logits = go_model.apply[VALUE_INDEX](params, rng_key, embeds)
        return ValueOutput(value=get_tromp_taylor_score(value_logits))

    return value_fn


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
            sampled_logits_plus_gumbel, sampled_actions = jax.lax.top_k(
                policy_logits + gumbel, k=sample_action_size)
            chex.assert_shape(sampled_actions,
                              (batch_size, sample_action_size))
            # N x A' x D x B x B
            partial_transitions = go_model.apply[TRANSITION_INDEX](
                params, rng_key, embeds, batch_partial_actions=sampled_actions)
            chex.assert_shape(
                partial_transitions,
                (batch_size, sample_action_size, hdim, board_size, board_size))
            flattened_partial_transiion_final_area_logits = go_model.apply[
                VALUE_INDEX](
                    params, rng_key,
                    nt_utils.flatten_first_two_dims(partial_transitions))
            partial_transition_value_logits = nt_utils.unflatten_first_dim(
                get_tromp_taylor_score(
                    flattened_partial_transiion_final_area_logits), batch_size,
                sample_action_size)
            chex.assert_shape(partial_transition_value_logits,
                              (batch_size, sample_action_size))
            # We take the negative of the transition logits because they're in
            # the opponent's perspective.
            qvals = -partial_transition_value_logits
            argmax_of_top_m = jnp.argmax(sampled_logits_plus_gumbel +
                                         scale_q_complete(qvals),
                                         axis=1)
            return PolicyOutput(sampled_actions=sampled_actions[
                jnp.arange(len(sampled_actions)),
                argmax_of_top_m].astype('uint16'),
                                visited_actions=None,
                                visited_qvalues=None)

    return policy_fn


def save_model(params: optax.Params,
               model_build_config: _build_config.ModelBuildConfig,
               model_dir: str):
    """Saves the parameters and build config into the directory."""
    if not drive.directory_exists(model_dir):
        drive.mkdir(model_dir)
    drive.write_file(
        os.path.join(model_dir, 'params.npz'),
        mode='wb',
        mime_type='application/octet-stream',
        write_fn=lambda fp: pickle.dump(
            jax.tree_map(lambda x: x.astype('float32'), params), fp))
    drive.write_file(os.path.join(model_dir, 'build_config.json'),
                     mode='wt',
                     mime_type='application/json',
                     write_fn=lambda fp: json.dump(
                         dataclasses.asdict(model_build_config), fp))
    logger.log(f"Saved model to '{model_dir}'.")


def load_model(
    load_dir: str
) -> Tuple[hk.MultiTransformed, optax.Params, _build_config.ModelBuildConfig]:
    """Loads the model from the given directory.

    Expects there to be one config.json file for the ModelBuildConfig
    and a params.npz file for the parameters.

    Args:
        load_dir (str): Model directory.

    Returns:
        Go model, parameters, and build config.
    """

    with drive.open_file(os.path.join(load_dir, 'build_config.json'),
                         mode='rt',
                         encoding='utf-8') as config_fp:
        json_dict = json.load(config_fp)
        meta_build_config = _build_config.MetaBuildConfig(
            **json_dict['meta_build_config'])
        model_build_config = _build_config.ModelBuildConfig(
            meta_build_config=meta_build_config,
            embed_build_config=_build_config.ComponentBuildConfig(
                **json_dict['embed_build_config']),
            area_build_config=_build_config.ComponentBuildConfig(
                **json_dict['area_build_config']),
            value_build_config=_build_config.ComponentBuildConfig(
                **json_dict['value_build_config']),
            policy_build_config=_build_config.ComponentBuildConfig(
                **json_dict['policy_build_config']),
            transition_build_config=_build_config.ComponentBuildConfig(
                **json_dict['transition_build_config']),
        )

    with drive.open_file(os.path.join(load_dir, 'params.npz'),
                         'rb') as file_array:
        params = pickle.load(file_array)
    go_model = _build_model_transform(model_build_config)
    return go_model, params, model_build_config
