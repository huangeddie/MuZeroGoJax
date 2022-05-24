"""
Models that map Go states to other vector spaces, which can be used for feature extraction or
dimensionality reduction.
"""

import gojax
import jax.numpy as jnp

from models import base


class Identity(base.BaseGoModel):
    """Identity model. Should be used with the real transition."""

    def __call__(self, states):
        return states


class BlackPerspective(base.BaseGoModel):
    """Converts all states whose turn is white to black's perspective."""

    def __call__(self, states):
        return jnp.where(jnp.expand_dims(gojax.get_turns(states), (1, 2, 3)),
                         gojax.swap_perspectives(states), states)
