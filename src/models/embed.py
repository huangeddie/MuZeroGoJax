"""
Models that map Go states to other vector spaces, which can be used for feature extraction or
dimensionality reduction.
"""

from models import base


class Identity(base.BaseGoModel):
    """Identity model. Should be used with the real transition."""

    def __call__(self, states):
        return states
