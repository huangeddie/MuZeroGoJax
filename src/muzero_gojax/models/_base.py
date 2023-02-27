"""All Go modules should subclass this module."""
from typing import Sequence, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from absl import flags

from muzero_gojax.models import _build_config

_BROADCAST_BIAS = flags.DEFINE_bool(
    "broadcast_bias", False, "Have bias in the ResNet broadcast linear layer.")

FloatStrBoolOrTuple = Union[str, float, bool, tuple]


class BaseGoModel(hk.Module):
    """All Go modules should subclass this module."""

    def __init__(self, model_config: _build_config.ModelBuildConfig,
                 submodel_config: _build_config.SubModelBuildConfig, *args,
                 **kwargs):
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


class Broadcast2D(hk.Module):
    """Mixes the features across the spatial dimensions (height, width)."""

    def __call__(self, input_3d: jnp.ndarray) -> jnp.ndarray:
        batch_size, channels, height, width = input_3d.shape
        out = input_3d.reshape((batch_size, channels, height * width))
        out = hk.Linear(height * width, with_bias=False, name='broadcast')(out)
        out = hk.LayerNorm(name="broadcast_layernorm",
                           axis=(1, 2, 3),
                           create_scale=True,
                           create_offset=True)(out)
        return jax.nn.relu(out)


class DpConvLnRl(hk.Module):
    """Dropout -> Conv -> LayerNorm -> ReLU."""

    def __init__(self, output_channels: int,
                 kernel_shape: Union[int, Sequence[int]], **kwargs):
        super().__init__(**kwargs)
        self._conv = hk.Conv2D(output_channels,
                               kernel_shape=kernel_shape,
                               data_format='NCHW',
                               padding="SAME")
        self._layer_norm = hk.LayerNorm(axis=(1, 2, 3),
                                        create_scale=True,
                                        create_offset=True)

    def __call__(self, input_3d: jnp.ndarray) -> jnp.ndarray:
        out = input_3d
        out = hk.dropout(hk.next_rng_key(), np.float16(0.1), out)
        out = self._conv(out)
        out = self._layer_norm(out)
        return jax.nn.relu(out)


class ResNetBlockV3(hk.Module):
    """My simplified version of ResNet V2 block."""

    def __init__(self, output_channels: int, hidden_channels: int, **kwargs):
        super().__init__(**kwargs)
        self._feature_conv = DpConvLnRl(output_channels=hidden_channels,
                                        kernel_shape=3)
        self._projection = DpConvLnRl(output_channels=output_channels,
                                      kernel_shape=1)

    def __call__(self, input_3d: jnp.ndarray) -> jnp.ndarray:
        out = input_3d
        out = self._feature_conv(out)
        out = self._projection(out)
        return out + input_3d


class ResNetBlockV2(hk.Module):
    """ResNet V2 block with LayerNorm and optional bottleneck."""

    def __init__(self,
                 channels: int,
                 stride: Union[int, Sequence[int]] = 1,
                 use_projection: bool = False,
                 broadcast: bool = False,
                 bottleneck_div: int = 4,
                 **kwargs):
        super().__init__(**kwargs)
        self.use_projection = use_projection
        self.broadcast = broadcast
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
        if self.broadcast:
            self.broadcast_ln = hk.LayerNorm(name="broadcast_layernorm",
                                             **ln_config)

        conv_0 = hk.Conv2D(data_format='NCHW',
                           output_channels=channels // bottleneck_div,
                           kernel_shape=1,
                           stride=1,
                           with_bias=False,
                           padding="SAME",
                           name="conv_0")

        ln_0 = hk.LayerNorm(name="layernorm_0", **ln_config)

        conv_1 = hk.Conv2D(data_format='NCHW',
                           output_channels=channels // bottleneck_div,
                           kernel_shape=3,
                           stride=stride,
                           with_bias=False,
                           padding="SAME",
                           name="conv_1")

        ln_1 = hk.LayerNorm(name="layernorm_1", **ln_config)
        layers = ((conv_0, ln_0), (conv_1, ln_1))

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
            if i == 1 and self.broadcast:
                batch_size, channels, height, width = out.shape
                out = out.reshape((batch_size, channels, height * width))
                out = hk.Linear(height * width,
                                with_bias=_BROADCAST_BIAS.value,
                                name='broadcast')(out)
                out = self.broadcast_ln(out)
                out = jax.nn.relu(out)
                out = out.reshape((batch_size, channels, height, width))
            out = conv_i(out)

        return out + shortcut


class ResNetV2(hk.Module):
    """ResNet model with dynamic layers.

    Ends with a normalization and ReLU."""

    def __init__(self,
                 hdim,
                 nlayers,
                 odim,
                 broadcast_frequency: int = 0,
                 bottleneck_div: int = 4,
                 **kwargs):
        super().__init__(**kwargs)

        self._initial_conv = hk.Conv2D(hdim,
                                       kernel_shape=1,
                                       data_format='NCHW')
        self.blocks = []
        for i in range(1, nlayers + 1):
            self.blocks.append(
                ResNetBlockV2(channels=odim if i == nlayers else hdim,
                              use_projection=(i == nlayers),
                              broadcast=(broadcast_frequency > 0
                                         and i % broadcast_frequency == 0),
                              bottleneck_div=bottleneck_div,
                              **kwargs))
        self._final_layer_norm = hk.LayerNorm(axis=(1, 2, 3),
                                              create_scale=True,
                                              create_offset=True)

    def __call__(self, inputs):
        out = self._initial_conv(inputs)
        for block in self.blocks:
            out = block(out)
        return jax.nn.relu(self._final_layer_norm(out))
