"""All Go modules should subclass this module."""
import haiku as hk
import jax


class BaseGoModel(hk.Module):
    """All Go modules should subclass this module."""

    def __init__(self, board_size, hdim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.board_size = board_size
        self.hdim = hdim
        self.action_size = board_size ** 2 + 1


class SimpleConvBlock(hk.Module):
    """Convolution -> Layer Norm -> ReLU -> Convolution."""

    def __init__(self, hdim, odim, use_layer_norm=True, **kwargs):
        super().__init__(**kwargs)
        self._conv1 = hk.Conv2D(hdim, (3, 3), data_format='NCHW')
        self._conv2 = hk.Conv2D(odim, (3, 3), data_format='NCHW')
        if use_layer_norm:
            self._maybe_layer_norm = hk.LayerNorm(axis=(1, 2, 3), create_scale=False,
                                                  create_offset=False)
        else:
            self._maybe_layer_norm = lambda x: x

    def __call__(self, input_3d):
        out = input_3d
        out = self._conv1(out)
        out = self._maybe_layer_norm(out)
        out = jax.nn.relu(out)
        out = self._conv2(out)
        out = self._maybe_layer_norm(out)
        return out
