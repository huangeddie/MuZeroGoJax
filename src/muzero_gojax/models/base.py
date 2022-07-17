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

    def __init__(self, hdim, odim, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self._conv1 = hk.Conv2D(hdim, (kernel_size, kernel_size), data_format='NCHW', name='conv1')
        self._layer_norm = hk.LayerNorm(axis=(1, 2, 3), create_scale=False, create_offset=False)
        self._conv2 = hk.Conv2D(odim, (kernel_size, kernel_size), data_format='NCHW', name='conv2')

    def __call__(self, input_3d):
        return self._layer_norm(
            self._conv2(jax.nn.relu(self._layer_norm(self._conv1(input_3d.astype('bfloat16'))))))
