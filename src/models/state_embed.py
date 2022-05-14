"""
Models that map Go states to other vector spaces, which can be used for feature extraction or
dimensionality reduction.
"""

from models import base_go_model


class StateIdentity(base_go_model.BaseGoModel):
    """Identity model. Should be used with the real transition."""

    def __call__(self, states):
        return states
