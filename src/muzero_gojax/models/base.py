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

    def __init__(self, hdim, odim, use_batch_norm=False, **kwargs):
        super().__init__(**kwargs)
        self._conv1 = hk.Conv2D(hdim, (3, 3), data_format='NCHW')
        self._conv2 = hk.Conv2D(odim, (3, 3), data_format='NCHW')
        if use_batch_norm:
            self._maybe_batch_norm = hk.BatchNorm(axis=(1, 2, 3), create_scale=False, create_offset=False)
        else:
            self._maybe_batch_norm = lambda x: x

    def __call__(self, input_3d):
        x = input_3d
        x = self._conv1(x)
        x = self._maybe_batch_norm(x)
        x = jax.nn.relu(x)
        x = self._conv2(x)
        x = self._maybe_batch_norm(x)
        return x
