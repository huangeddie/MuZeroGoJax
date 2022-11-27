"""All Go modules should subclass this module."""
from typing import Sequence, Tuple, Union

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from absl import flags

_BOTTLENECK_RESNET = flags.DEFINE_bool(
    "bottleneck_resnet", False,
    "Whether or not to apply the ResNet bottleneck technique.")


@chex.dataclass(frozen=True)
class ModelBuildConfig:
    """Build config for whole Go model."""
    board_size: int = -1
    hdim: int = -1
    embed_dim: int = -1
    dtype: str = None


@chex.dataclass(frozen=True)
class SubModelBuildConfig:
    """Build config for submodel."""
    name_key: str = None
    nlayers: int = -1
    model_build_config: ModelBuildConfig = ModelBuildConfig()


FloatStrBoolOrTuple = Union[str, float, bool, tuple]


class BaseGoModel(hk.Module):
    """All Go modules should subclass this module."""

    def __init__(self, model_config: ModelBuildConfig,
                 submodel_config: SubModelBuildConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_config = model_config
        self.submodel_config = submodel_config

    def implicit_action_size(self, embeds: jnp.ndarray) -> Tuple:
        """Returns implicit action size from the embeddings."""
        return embeds.shape[-2] * embeds.shape[-1] + 1


class NonSpatialConv(hk.Module):
    """1x1 convolutions."""

    def __init__(self, hdim, odim, nlayers, **kwargs):
        super().__init__(**kwargs)
        self.convs = []
        for _ in range(max(nlayers, 0)):
            self.convs.append(hk.Conv2D(hdim, (1, 1), data_format='NCHW'))
        self._final_conv = hk.Conv2D(odim, (1, 1), data_format='NCHW')

    def __call__(self, input_3d):
        out = input_3d
        for conv in self.convs:
            out = conv(out)
            out = jax.nn.relu(out)
        return self._final_conv(out)


class ResNetBlockV2(hk.Module):
    """ResNet V2 block with LayerNorm and optional bottleneck."""

    def __init__(self,
                 channels: int,
                 stride: Union[int, Sequence[int]] = 1,
                 use_projection: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.use_projection = use_projection
        ln_config = {
            'axis': (1, 2, 3),
            'create_scale': True,
            'create_offset': True
        }

        if self.use_projection:
            self.proj_conv = hk.Conv2D(data_format='NCHW',
                                       output_channels=channels,
                                       kernel_shape=1,
                                       stride=stride,
                                       with_bias=False,
                                       padding="SAME",
                                       name="shortcut_conv")

        channel_div = 4 if _BOTTLENECK_RESNET.value else 1
        conv_0 = hk.Conv2D(data_format='NCHW',
                           output_channels=channels // channel_div,
                           kernel_shape=1 if _BOTTLENECK_RESNET.value else 3,
                           stride=1 if _BOTTLENECK_RESNET.value else stride,
                           with_bias=False,
                           padding="SAME",
                           name="conv_0")

        ln_0 = hk.LayerNorm(name="layernorm_0", **ln_config)

        conv_1 = hk.Conv2D(data_format='NCHW',
                           output_channels=channels // channel_div,
                           kernel_shape=3,
                           stride=stride if _BOTTLENECK_RESNET.value else 1,
                           with_bias=False,
                           padding="SAME",
                           name="conv_1")

        ln_1 = hk.LayerNorm(name="layernorm_1", **ln_config)
        layers = ((conv_0, ln_0), (conv_1, ln_1))

        if _BOTTLENECK_RESNET.value:
            conv_2 = hk.Conv2D(data_format='NCHW',
                               output_channels=channels,
                               kernel_shape=1,
                               stride=1,
                               with_bias=False,
                               padding="SAME",
                               name="conv_2")

            # NOTE: Some implementations of ResNet50 v2 suggest initializing
            # gamma/scale here to zeros.
            ln_2 = hk.LayerNorm(name="layernorm_2", **ln_config)
            layers = layers + ((conv_2, ln_2), )

        self.layers = layers

    def __call__(self, inputs):
        out = shortcut = inputs

        for i, (conv_i, ln_i) in enumerate(self.layers):
            out = ln_i(out)
            out = jax.nn.relu(out)
            if i == 0 and self.use_projection:
                shortcut = self.proj_conv(out)
            out = conv_i(out)

        return out + shortcut


class ResNetV2(hk.Module):
    """ResNet model with dynamic layers."""

    def __init__(self, hdim, nlayers, odim, **kwargs):
        super().__init__(**kwargs)

        self._initial_conv = hk.Conv2D(hdim, (3, 3), data_format='NCHW')
        self.blocks = []
        for _ in range(nlayers - 1):
            self.blocks.append(ResNetBlockV2(channels=hdim, **kwargs))
        self.blocks.append(
            ResNetBlockV2(channels=odim, use_projection=True, **kwargs))
        self._final_layer_norm = hk.LayerNorm(axis=(1, 2, 3),
                                              create_scale=True,
                                              create_offset=True)

    def __call__(self, inputs):
        out = self._initial_conv(inputs)
        for block in self.blocks:
            out = block(out)
        return jax.nn.relu(self._final_layer_norm(out))
