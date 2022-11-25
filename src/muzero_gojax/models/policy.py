"""Models that map state embeddings to state-action policy logits."""

import gojax
import haiku as hk
import jax
import jax.numpy as jnp

from muzero_gojax.models import base


class RandomPolicy(base.BaseGoModel):
    """Outputs independent standard normal variables."""

    def __call__(self, embeds):
        return jax.random.normal(
            hk.next_rng_key(),
            (len(embeds), self.implicit_action_size(embeds)))


class Linear3DPolicy(base.BaseGoModel):
    """Linear model."""

    def __call__(self, embeds):
        embeds = embeds.astype(self.model_config.dtype)
        action_w = hk.get_parameter('action_w',
                                    shape=(*embeds.shape[1:],
                                           self.implicit_action_size(embeds)),
                                    init=hk.initializers.RandomNormal(
                                        1. / self.model_config.board_size))
        action_b = hk.get_parameter('action_b',
                                    shape=(1,
                                           self.implicit_action_size(embeds)),
                                    init=jnp.zeros)

        return jnp.einsum('bchw,chwa->ba', embeds, action_w) + action_b


class NonSpatialConvPolicy(base.BaseGoModel):
    """Linear convolution model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._action_conv = base.NonSpatialConv(
            hdim=self.model_config.hdim,
            odim=1,
            nlayers=self.submodel_config.nlayers)
        self._pass_conv = base.NonSpatialConv(
            hdim=self.model_config.hdim,
            odim=1,
            nlayers=self.submodel_config.nlayers)

    def __call__(self, embeds):
        embeds = embeds.astype(self.model_config.dtype)
        move_logits = self._action_conv(embeds)
        pass_logits = jnp.expand_dims(jnp.mean(self._pass_conv(embeds),
                                               axis=(1, 2, 3)),
                                      axis=1)
        return jnp.concatenate(
            (jnp.reshape(move_logits,
                         (len(embeds), self.implicit_action_size(embeds) - 1)),
             pass_logits),
            axis=1)


class LinearConvPolicy(base.BaseGoModel):
    """Non-spatial linear convolution model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._action_conv = hk.Conv2D(1, (1, 1), data_format='NCHW')
        self._pass_conv = hk.Conv2D(1, (1, 1), data_format='NCHW')

    def __call__(self, embeds):
        embeds = embeds.astype(self.model_config.dtype)
        move_logits = self._action_conv(embeds)
        pass_logits = jnp.expand_dims(jnp.mean(self._pass_conv(embeds),
                                               axis=(1, 2, 3)),
                                      axis=1)
        return jnp.concatenate(
            (jnp.reshape(move_logits,
                         (len(embeds), self.implicit_action_size(embeds) - 1)),
             pass_logits),
            axis=1)


class SingleLayerConvPolicy(base.BaseGoModel):
    """LayerNorm -> ReLU -> Conv."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._layer_norm = hk.LayerNorm(axis=(1, 2, 3),
                                        create_scale=True,
                                        create_offset=True)
        self._action_conv = base.NonSpatialConv(hdim=self.model_config.hdim,
                                                odim=1,
                                                nlayers=1)
        self._pass_conv = base.NonSpatialConv(hdim=self.model_config.hdim,
                                              odim=1,
                                              nlayers=1)

    def __call__(self, embeds):
        out = embeds.astype(self.model_config.dtype)
        out = self._layer_norm(out)
        out = jax.nn.relu(out)
        move_logits = self._action_conv(out)
        pass_logits = jnp.expand_dims(jnp.mean(self._pass_conv(out),
                                               axis=(1, 2, 3)),
                                      axis=1)
        return jnp.concatenate((jnp.reshape(
            move_logits,
            (len(out), self.implicit_action_size(out) - 1)), pass_logits),
                               axis=1)


class ResNetV2Policy(base.BaseGoModel):
    """ResNetV2 model."""

    def __init__(self, *args, **kwargs):
        # pylint: disable=duplicate-code
        super().__init__(*args, **kwargs)
        self._resnet = base.ResNetV2(hdim=self.model_config.hdim,
                                     nlayers=self.submodel_config.nlayers,
                                     odim=self.model_config.hdim)
        self._final_action_conv = hk.Conv2D(1, (1, 1), data_format='NCHW')
        self._final_pass_conv = hk.Conv2D(1, (1, 1), data_format='NCHW')

    def __call__(self, embeds):
        out = self._resnet(embeds.astype(self.model_config.dtype))
        action_out = self._final_action_conv(out)
        pass_out = jnp.expand_dims(jnp.mean(self._final_pass_conv(out),
                                            axis=(1, 2, 3)),
                                   axis=1)
        return jnp.concatenate((jnp.reshape(
            action_out,
            (len(embeds), self.implicit_action_size(embeds) - 1)), pass_out),
                               axis=1)


class TrompTaylorPolicy(base.BaseGoModel):
    """
    Logits equal to player's area - opponent's area for next state.

    Requires identity embedding.
    """

    def __call__(self, embeds):
        states = embeds.astype(bool)
        all_children = gojax.get_children(states)
        batch_size, action_size, channels, nrows, ncols = all_children.shape
        turns = jnp.repeat(jnp.expand_dims(gojax.get_turns(states), axis=1),
                           repeats=action_size,
                           axis=1)
        flat_children = jnp.reshape(
            all_children, (batch_size * action_size, channels, nrows, ncols))
        flat_turns = jnp.reshape(turns, batch_size * action_size)
        sizes = gojax.compute_area_sizes(flat_children).astype(
            self.model_config.dtype)
        n_idcs = jnp.arange(len(sizes))
        return jnp.reshape(
            sizes[n_idcs, flat_turns.astype('uint8')] -
            sizes[n_idcs,
                  (~flat_turns).astype('uint8')], (batch_size, action_size))


class TrompTaylorAmplifiedPolicy(TrompTaylorPolicy):
    """
    Logits equal to (player's area - opponent's area) * 100 for next state.

    Requires identity embedding.
    """

    def __call__(self, embeds):
        return super().__call__(embeds) * jnp.array(
            100, dtype=self.model_config.dtype)
