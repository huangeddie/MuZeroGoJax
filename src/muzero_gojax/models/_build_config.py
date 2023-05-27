"""Model build configs."""

import chex
from absl import flags

_EMBED_MODEL = flags.DEFINE_string(
    'embed_model', 'LinearConvEmbed', 'Class name of the submodel to use. '
    'See the submodel module to view all submodel classes.')
_AREA_MODEL = flags.DEFINE_string(
    'area_model', 'LinearConvArea', 'Class name of the submodel to use. '
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
_BOTTLENECK_DIV = flags.DEFINE_integer(
    "bottleneck_div", 4,
    "How much to divide the channels by in the ResNet bottleneck layers.")
_BROADCAST_FREQUENCY = flags.DEFINE_integer(
    'broadcast_frequency', 0, 'Broadcast '
    'frequency for ResNet blocks.')
_EMBED_NLAYERS = flags.DEFINE_integer('embed_nlayers', 0,
                                      'Number of embed layers.')
_VALUE_NLAYERS = flags.DEFINE_integer('value_nlayers', 0,
                                      'Number of value layers.')
_AREA_NLAYERS = flags.DEFINE_integer('area_nlayers', 0,
                                     'Number of area layers.')
_POLICY_NLAYERS = flags.DEFINE_integer('policy_nlayers', 0,
                                       'Number of policy layers.')
_TRANSITION_NLAYERS = flags.DEFINE_integer('transition_nlayers', 0,
                                           'Number of transition layers.')


@chex.dataclass(frozen=True)
class MetaBuildConfig:
    """Build config for whole Go model."""
    board_size: int = -1
    hdim: int = -1
    embed_dim: int = -1
    # Applies broadcasting every n'th layer. 0 means no broadcasting.
    broadcast_frequency: int = 0
    # How much to divide the channels by in the ResNet bottleneck layers.
    bottleneck_div: int = 4


@chex.dataclass(frozen=True)
class ComponentBuildConfig:
    """Build config for submodel."""
    name_key: str = None
    nlayers: int = -1


@chex.dataclass(frozen=True)
class ModelBuildConfig:
    """All model and submodel build configs.

    This config is serializable to JSON.
    """
    meta_build_config: MetaBuildConfig
    embed_build_config: ComponentBuildConfig
    area_build_config: ComponentBuildConfig
    value_build_config: ComponentBuildConfig
    policy_build_config: ComponentBuildConfig
    transition_build_config: ComponentBuildConfig


def get_model_build_config(board_size: int) -> ModelBuildConfig:
    """Returns all the model configs from the flags."""
    meta_build_config = MetaBuildConfig(
        board_size=board_size,
        hdim=_HDIM.value,
        embed_dim=_EMBED_DIM.value,
        broadcast_frequency=_BROADCAST_FREQUENCY.value,
        bottleneck_div=_BOTTLENECK_DIV.value)
    embed_build_config = ComponentBuildConfig(name_key=_EMBED_MODEL.value,
                                              nlayers=_EMBED_NLAYERS.value)
    area_build_config = ComponentBuildConfig(name_key=_AREA_MODEL.value,
                                             nlayers=_AREA_NLAYERS.value)
    value_build_config = ComponentBuildConfig(name_key=_VALUE_MODEL.value,
                                              nlayers=_VALUE_NLAYERS.value)
    policy_build_config = ComponentBuildConfig(name_key=_POLICY_MODEL.value,
                                               nlayers=_POLICY_NLAYERS.value)
    transition_build_config = ComponentBuildConfig(
        name_key=_TRANSITION_MODEL.value, nlayers=_TRANSITION_NLAYERS.value)
    return ModelBuildConfig(meta_build_config=meta_build_config,
                            embed_build_config=embed_build_config,
                            area_build_config=area_build_config,
                            value_build_config=value_build_config,
                            policy_build_config=policy_build_config,
                            transition_build_config=transition_build_config)
