"""All Go modules should subclass this module."""
import haiku as hk
import jax


class BaseGoModel(hk.Module):
    """All Go modules should subclass this module."""

    def __init__(self, board_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.board_size = board_size
        self.action_size = board_size ** 2 + 1


class SimpleConvBlock(hk.Module):
    """Black perspective embedding followed by a light-weight CNN neural network."""

    def __init__(self, hdim, odim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv1 = hk.Conv2D(hdim, (3, 3), data_format='NCHW')
        self._layer_norm = hk.LayerNorm(axis=(1, 2, 3), create_scale=False, create_offset=False)
        self._conv2 = hk.Conv2D(odim, (3, 3), data_format='NCHW')

    def __call__(self, input_3d):
        return self._layer_norm(
            self._conv2(jax.nn.relu(self._layer_norm(self._conv1(input_3d.astype('bfloat16'))))))
