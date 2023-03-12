"""Models that map state embeddings to the next state embeddings for all actions."""

from typing import Tuple

import gojax
import haiku as hk
import jax
import jax.nn
import jax.numpy as jnp
from jax import lax

from muzero_gojax import nt_utils
from muzero_gojax.models import _base, _embed


class BaseTransitionModel(_base.BaseGoModel):
    """Adds more transition helper functions."""

    def implicit_transition_output_shape(self, embeds: jnp.ndarray) -> Tuple:
        """Returns implicit transition output shape assuming the embeddings
        preserves the board size as the height and width."""
        return (len(embeds), self.implicit_action_size(embeds),
                *embeds.shape[1:])

    def partial_action_transition_output_shape(
            self, embeds: jnp.ndarray,
            partial_action_size: jnp.ndarray) -> Tuple:
        """Returns transition output shape with a partial action size."""
        return (len(embeds), partial_action_size, *embeds.shape[1:])

    def get_partial_action_size(self,
                                batch_partial_actions: jnp.ndarray) -> int:
        """Returns the batch partial action size."""
        return batch_partial_actions.shape[1]

    def default_all_actions(self, embeds: jnp.ndarray) -> jnp.ndarray:
        """Returns all actions per embeddings.

        Args:
            embeds (jnp.ndarray): N x D x B x B

        Returns:
            jnp.ndarray: N x A
        """
        batch_size = len(embeds)
        batch_partial_actions = jnp.repeat(jnp.expand_dims(jnp.arange(
            self.implicit_action_size(embeds)),
                                                           axis=0),
                                           repeats=batch_size,
                                           axis=0)

        return batch_partial_actions

    def embed_actions(self, embeds: jnp.ndarray,
                      batch_partial_actions: jnp.ndarray) -> jnp.ndarray:
        """Returns the embeddings with the actions embedded as indicator actions.

        Args:
            embeds (jnp.ndarray): N x D x B x B
            batch_partial_actions (jnp.ndarray): N x A'

        Returns:
            jnp.ndarray: N x A' x (D+1) x B x B
        """
        board_size = embeds.shape[-1]
        partial_action_size = batch_partial_actions.shape[1]
        batch_indicator_actions = nt_utils.unflatten_first_dim(
            gojax.action_1d_to_indicator(
                nt_utils.flatten_first_two_dims(batch_partial_actions),
                board_size, board_size), *batch_partial_actions.shape[:2])

        # N x A' x (D+1) x B x B
        duplicated_embeds = jnp.repeat(jnp.expand_dims(embeds.astype(
            self.model_config.dtype),
                                                       axis=1),
                                       repeats=partial_action_size,
                                       axis=1)
        embeds_with_actions = jnp.concatenate(
            (duplicated_embeds, jnp.expand_dims(batch_indicator_actions,
                                                axis=2)),
            axis=2)

        return embeds_with_actions


class RandomTransition(BaseTransitionModel):
    """Outputs independent standard normal variables."""

    def __call__(self, embeds, batch_partial_actions: jnp.ndarray = None):
        if batch_partial_actions is None:
            batch_partial_actions = self.default_all_actions(embeds)
        return jax.random.normal(
            hk.next_rng_key(),
            self.partial_action_transition_output_shape(
                embeds, self.get_partial_action_size(batch_partial_actions)),
            dtype=self.model_config.dtype)


class RealTransition(BaseTransitionModel):
    """
    Real Go transitions.

    Should be used with the identity embedding.
    """

    def __call__(self, embeds, batch_partial_actions: jnp.ndarray = None):
        if batch_partial_actions is None:
            batch_partial_actions = self.default_all_actions(embeds)
        return lax.stop_gradient(
            gojax.expand_states(embeds.astype(bool),
                                batch_partial_actions).astype(
                                    self.model_config.dtype))


class BlackRealTransition(BaseTransitionModel):
    """
    Real Go transitions under black's perspective.

    Should be used with the BlackPerspectiveEmbed embedding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._internal_real_transition = RealTransition(*args, **kwargs)
        self._internal_canonical_embed = _embed.CanonicalEmbed(*args, **kwargs)

    def __call__(self, embeds, batch_partial_actions: jnp.ndarray = None):
        if batch_partial_actions is None:
            batch_partial_actions = self.default_all_actions(embeds)
        transitions = self._internal_real_transition(embeds)
        batch_size, action_size, channel, board_height, board_width = transitions.shape
        canonicals = self._internal_canonical_embed(
            jnp.reshape(transitions.astype(bool),
                        (batch_size * action_size, channel, board_height,
                         board_width)))
        return lax.stop_gradient(jnp.reshape(canonicals, transitions.shape))


class LinearConvTransition(BaseTransitionModel):
    """Linear model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv = hk.Conv2D(self.model_config.embed_dim, (1, 1),
                               data_format='NCHW')

    def __call__(self, embeds, batch_partial_actions: jnp.ndarray = None):
        # Embeds is N x D x B x B
        # N x A' x B x B
        if batch_partial_actions is None:
            batch_partial_actions = self.default_all_actions(embeds)
        embeds_with_actions = self.embed_actions(embeds, batch_partial_actions)

        partial_action_size = self.get_partial_action_size(
            batch_partial_actions)

        # N x A' x (D*)
        return nt_utils.unflatten_first_dim(
            self._conv(nt_utils.flatten_first_two_dims(embeds_with_actions)),
            len(embeds), partial_action_size)


class NonSpatialConvTransition(BaseTransitionModel):
    """Linear model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv = _base.NonSpatialConv(hdim=self.model_config.hdim,
                                          odim=self.model_config.embed_dim,
                                          nlayers=0)

    def __call__(self, embeds, batch_partial_actions: jnp.ndarray = None):
        # Embeds is N x D x B x B
        # N x A' x B x B
        if batch_partial_actions is None:
            batch_partial_actions = self.default_all_actions(embeds)
        embeds_with_actions = self.embed_actions(embeds, batch_partial_actions)

        partial_action_size = self.get_partial_action_size(
            batch_partial_actions)

        # N x A' x (D*)
        return nt_utils.unflatten_first_dim(
            self._conv(nt_utils.flatten_first_two_dims(embeds_with_actions)),
            len(embeds), partial_action_size)


class ResNetV2Transition(BaseTransitionModel):
    """ResNetV2 model."""

    def __init__(self, *args, **kwargs):
        # pylint: disable=duplicate-code
        super().__init__(*args, **kwargs)
        self._resnet = _base.ResNetV2(
            hdim=self.model_config.hdim,
            nlayers=self.submodel_config.nlayers,
            odim=self.model_config.hdim,
            bottleneck_div=self.model_config.bottleneck_div)
        self._conv = hk.Conv2D(self.model_config.embed_dim, (1, 1),
                               data_format='NCHW')

    def __call__(self,
                 embeds: jnp.ndarray,
                 batch_partial_actions: jnp.ndarray = None) -> jnp.ndarray:
        """Inference transition model by embedding indicator actions into embeddings.

        If batch_partial_actions is specified, it inferences only those
        specified actions and all other transitions are zeros.

        Args:
            embeds (jnp.ndarray): N x D x B x B
            batch_partial_actions (jnp.ndarray, optional): N x A'. Defaults to None.

        Returns:
            jnp.ndarray: N x A x D x B x B
        """
        # Embeds is N x D x B x B
        # N x A' x B x B
        if batch_partial_actions is None:
            batch_partial_actions = self.default_all_actions(embeds)
        embeds_with_actions = self.embed_actions(embeds, batch_partial_actions)

        partial_action_size = self.get_partial_action_size(
            batch_partial_actions)

        # N x A' x (D*)
        batch_size = len(embeds)
        return nt_utils.unflatten_first_dim(
            self._conv(
                self._resnet(
                    nt_utils.flatten_first_two_dims(embeds_with_actions))),
            batch_size, partial_action_size)


class ResNetV3Transition(BaseTransitionModel):
    """My simplified version of ResNet V2."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._blocks = [
            _base.DpConvLnRl(output_channels=256, kernel_shape=1),
            _base.ResNetBlockV3(output_channels=self.model_config.embed_dim,
                                hidden_channels=self.model_config.hdim),
            _base.ResNetBlockV3(output_channels=self.model_config.embed_dim,
                                hidden_channels=self.model_config.hdim),
            _base.ResNetBlockV3(output_channels=self.model_config.embed_dim,
                                hidden_channels=self.model_config.hdim),
            _base.ResNetBlockV3(output_channels=self.model_config.embed_dim,
                                hidden_channels=self.model_config.hdim),
        ]

    def __call__(self,
                 embeds: jnp.ndarray,
                 batch_partial_actions: jnp.ndarray = None) -> jnp.ndarray:
        """Inference transition model by embedding indicator actions into embeddings.

        If batch_partial_actions is specified, it inferences only those
        specified actions and all other transitions are zeros.

        Args:
            embeds (jnp.ndarray): N x D x B x B
            batch_partial_actions (jnp.ndarray, optional): N x A'. Defaults to None.

        Returns:
            jnp.ndarray: N x A x D x B x B
        """
        # Embeds is N x D x B x B
        # N x A' x B x B
        if batch_partial_actions is None:
            batch_partial_actions = self.default_all_actions(embeds)
        embeds_with_actions = self.embed_actions(embeds, batch_partial_actions)

        out = nt_utils.flatten_first_two_dims(embeds_with_actions)
        for block in self._blocks:
            out = block(out)

        # N x A' x (D*)
        batch_size = len(embeds)
        partial_action_size = self.get_partial_action_size(
            batch_partial_actions)
        return nt_utils.unflatten_first_dim(out, batch_size,
                                            partial_action_size)
