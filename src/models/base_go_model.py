"""All Go modules should subclass this module."""
import haiku as hk


class BaseGoModel(hk.Module):
    """All Go modules should subclass this module."""

    def __init__(self, board_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.board_size = board_size
        self.action_size = board_size ** 2 + 1
