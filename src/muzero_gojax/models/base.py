"""All Go modules should subclass this module."""
from typing import NamedTuple
from typing import Sequence
from typing import Union

import haiku as hk
import jax
from absl import flags

_BOTTLENECK_RESNET = flags.DEFINE_bool(
    "bottleneck_resnet", False,
    "Whether or not to apply the ResNet bottleneck technique.")


class ModelBuildParams(NamedTuple):
    """Parameters to controlling how to build the model."""
    board_size: int = -1
    hdim: int = -1
    nlayers: int = -1
    embed_dim: int = -1
    dtype: str = None

    embed_model_key: str = None
    decode_model_key: str = None
    value_model_key: str = None
    policy_model_key: str = None
    transition_model_key: str = None


FloatStrBoolOrTuple = Union[str, float, bool, tuple]


class BaseGoModel(hk.Module):
    """All Go modules should subclass this module."""

    def __init__(self, model_params: ModelBuildParams, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_params = model_params
        self.action_size = self.model_params.board_size**2 + 1
        self.transition_output_shape = (-1, self.action_size,
                                        self.model_params.embed_dim,
                                        self.model_params.board_size,
                                        self.model_params.board_size)


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


class SimpleConvBlock(hk.Module):
    """Convolution -> Layer Norm -> ReLU -> Convolution."""

    def __init__(self, hdim, odim, use_layer_norm=True, **kwargs):
        super().__init__(**kwargs)
        self._conv1 = hk.Conv2D(hdim, (3, 3), data_format='NCHW')
        self._conv2 = hk.Conv2D(odim, (3, 3), data_format='NCHW')
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self._ln0 = hk.LayerNorm(axis=(1, 2, 3),
                                     create_scale=True,
                                     create_offset=True)
            self._ln1 = hk.LayerNorm(axis=(1, 2, 3),
                                     create_scale=True,
                                     create_offset=True)

    def __call__(self, input_3d):
        out = input_3d
        out = self._conv1(out)
        if self.use_layer_norm:
            out = self._ln0(out)
        out = jax.nn.relu(out)
        out = self._conv2(out)
        if self.use_layer_norm:
            out = self._ln1(out)
        return out


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
