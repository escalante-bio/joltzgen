"""

Simple translation of Boltz-1 to JAX.

We use single-dispatch to convert PyTorch modules to Equinox modules recursively.

At a high-level we use the single-dispatch function `from_torch` to transform PyTorch modules to Equinox modules -- we use almost exactly the same names for the Equinox modules and their properties.

I use the convention that each equinox module has a static (or class) `from_torch` method that takes as its sole argument the corresponding PyTorch module and returns an instance of the equinox class.
Usually, this method calls `from_torch` on all of the pytorch module's children, and then constructs the equinox module using the results.
Because the implementation of this function is almost always the same I use `AbstractFromTorch` to define a default implementation of `from_torch`.
The final step is to register that function with the `from_torch` dispatcher. This is done with the `register_from_torch` class decorator, which takes a PyTorch module type and returns a decorator that registers the `from_torch` method of the equinox module.

`backend.py` contains the basic machinery for this and some translations of vanilla PyTorch modules.
This file contains translations of Boltz-1 modules.


"""

from functools import partial
from typing import Literal

import boltzgen
import boltzgen.model.layers.outer_product_mean
import boltzgen.model.layers.pair_averaging
import boltzgen.model.layers.pairformer
import boltzgen.model.layers.transition
import boltzgen.model.layers.triangular
import boltzgen.model.layers.triangular_attention.attention
import boltzgen.model.layers.triangular_attention.primitives
import boltzgen.model.models
import boltzgen.model.models.boltz
import boltzgen.model.modules.confidence
import boltzgen.model.modules.diffusion
import boltzgen.model.modules.diffusion_conditioning
import boltzgen.model.modules.trunk
import boltzgen.model.modules.utils
import einops
import equinox as eqx
import jax
import numpy as np
from boltzgen.data import const
from jax import numpy as jnp
from jax import tree, vmap
from jaxtyping import Array, Bool, Float

from .backend import (
    AbstractFromTorch,
    Embedding,
    Identity,
    LayerNorm,
    Linear,
    Sequential,
    from_torch,
    register_from_torch,
)


@from_torch.register(boltzgen.model.modules.utils.SwiGLU)
def _handle(_):
    def _swiglu(x):
        x, gates = jnp.split(x, 2, axis=-1)
        return jax.nn.silu(gates) * x

    return _swiglu


def get_dropout_mask(
    dropout: float,
    z: Float[Array, "..."],
    training: bool,
    columnwise: bool = False,
    *,
    key,
) -> Float[Array, "..."]:
    dropout = dropout * training
    v = z[:, 0:1, :, 0:1] if columnwise else z[:, :, 0:1, 0:1]
    # d = torch.rand_like(v) > dropout
    d = jax.random.bernoulli(key, 1 - dropout, v.shape) > 0
    d = d * 1.0 / (1.0 - dropout)
    return d, jax.random.fold_in(key, 0)


@register_from_torch(boltzgen.model.layers.transition.Transition)
class Transition(AbstractFromTorch):
    norm: LayerNorm
    fc1: Linear
    fc2: Linear
    fc3: Linear
    silu: callable

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... P"]:
        v = self.norm(x)
        return self.fc3(jax.nn.silu(self.fc1(v)) * self.fc2(v))


@register_from_torch(boltzgen.model.layers.pair_averaging.PairWeightedAveraging)
class PairWeightedAveraging(AbstractFromTorch):
    norm_m: LayerNorm
    norm_z: LayerNorm
    proj_m: Linear
    proj_g: Linear
    proj_z: Linear
    proj_o: Linear
    num_heads: int
    c_h: int  # channel heads
    inf: float

    def __call__(
        self,
        m: Float[Array, "B S N D"],
        z: Float[Array, "B N N D"],
        mask: Bool[Array, "B N N"],
    ) -> Float[Array, "B S N D"]:
        m = self.norm_m(m)
        z = self.norm_z(z)

        # follows the pytorch implementation -- could be rewritten in a more jax'ian (jaxish?) style

        # Project input tensors
        v = self.proj_m(m)
        v = v.reshape(*v.shape[:3], self.num_heads, self.c_h)  # no batch dimension
        v = jnp.transpose(v, (0, 3, 1, 2, 4))  # move heads to front, (b h s i d)

        # Compute weights
        b = self.proj_z(z)
        b = jnp.transpose(b, (0, 3, 1, 2))  # (b h i j)
        b = b + (1 - mask[:, None]) * -self.inf
        w = jax.nn.softmax(b, axis=-1)
        # Compute gating
        g = self.proj_g(m)
        g = jax.nn.sigmoid(g)

        # Compute output
        o = jnp.einsum("bhij,bhsjd->bhsid", w, v)
        o = jnp.transpose(o, (0, 2, 3, 1, 4))
        o = o.reshape(*o.shape[:3], self.num_heads * self.c_h)
        return self.proj_o(g * o)


@register_from_torch(boltzgen.model.layers.triangular.TriangleMultiplicationOutgoing)
class TriangleMultiplicationOutgoing(AbstractFromTorch):
    norm_in: LayerNorm
    p_in: Linear
    g_in: Linear
    norm_out: LayerNorm
    p_out: Linear
    g_out: Linear

    def __call__(
        self, x: Float[Array, "N N D"], mask: Bool[Array, "N N"]
    ) -> Float[Array, "N N D"]:
        x = self.norm_in(x)
        x_in = x

        x = self.p_in(x) * jax.nn.sigmoid(self.g_in(x))

        x = x * mask[..., None]

        a, b = jnp.split(x, 2, axis=-1)

        x = jnp.einsum("bikd,bjkd->bijd", a, b)

        return self.p_out(self.norm_out(x)) * jax.nn.sigmoid(self.g_out(x_in))


@register_from_torch(boltzgen.model.layers.triangular.TriangleMultiplicationIncoming)
class TriangleMultiplicationIncoming(AbstractFromTorch):
    norm_in: LayerNorm
    p_in: Linear
    g_in: Linear
    norm_out: LayerNorm
    p_out: Linear
    g_out: Linear

    def __call__(
        self, x: Float[Array, "N N D"], mask: Bool[Array, "N N"]
    ) -> Float[Array, "N N D"]:
        x = self.norm_in(x)
        x_in = x

        x = self.p_in(x) * jax.nn.sigmoid(self.g_in(x))

        x = x * mask[..., None]

        a, b = jnp.split(x, 2, axis=-1)

        x = jnp.einsum("bkid,bkjd->bijd", a, b)

        return self.p_out(self.norm_out(x)) * jax.nn.sigmoid(self.g_out(x_in))


@register_from_torch(boltzgen.model.layers.triangular_attention.primitives.Attention)
class Attention(AbstractFromTorch):
    c_q: int  # input dimension of query
    c_k: int  # input dimension of key
    c_v: int  # input dimension of value
    c_hidden: int  # per-head hidden dimension
    no_heads: int  # number of heads
    gating: bool  # whether to use gating
    linear_q: Linear
    linear_k: Linear
    linear_v: Linear
    linear_o: Linear
    linear_g: Linear | None
    sigmoid: callable

    # TODO: Add mask? Instead of infs....
    def __call__(
        self,
        q_x: Float[Array, "... Q C_q"],
        kv_x: Float[Array, "... K C_k"],
        biases: None | list[Float[Array, "... H Q K"]],
    ) -> Float[Array, "... Q C_v"]:
        # apply linear
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)
        # and reshape rearrange to heads (..., H, Q/K/V, C_hidden)
        q = einops.rearrange(
            q, "... Q (H C_hidden) -> ... H Q C_hidden", H=self.no_heads
        )
        k = einops.rearrange(
            k, "... K (H C_hidden) -> ... H K C_hidden", H=self.no_heads
        )
        v = einops.rearrange(
            v, "... V (H C_hidden) -> ... H V C_hidden", H=self.no_heads
        )

        # scale q
        q = q / np.sqrt(self.c_hidden)

        # compute attention
        a = jnp.einsum("... h q d, ... h k d -> ... h q k", q, k)
        # add pairwise biases
        # todo: not this.
        for bias in biases:
            a += bias

        a = jax.nn.softmax(a, axis=-1)

        a = jnp.einsum("... h q k, ... h k d -> ... h q d", a, v)
        # equivalent of o = o.transpose(-2, -3)
        o = einops.rearrange(a, "... H Q C_hidden -> ... Q H C_hidden")
        if self.linear_g is not None:
            g = jax.nn.sigmoid(self.linear_g(q_x))
            g = einops.rearrange(
                g, "... (H C_hidden) -> ... H C_hidden", H=self.no_heads
            )
            o = o * g

        o = einops.rearrange(o, "... Q H C -> ... Q (H C)")

        return self.linear_o(o)


@from_torch.register(boltzgen.model.layers.triangular_attention.primitives.LayerNorm)
def _trilayer_norm(m: boltzgen.model.layers.triangular_attention.primitives.LayerNorm):
    assert len(m.c_in) == 1
    return LayerNorm(weight=from_torch(m.weight), bias=from_torch(m.bias), eps=m.eps)


@register_from_torch(
    boltzgen.model.layers.triangular_attention.attention.TriangleAttention
)
class TriangleAttention(AbstractFromTorch):
    c_in: int
    c_hidden: int
    no_heads: int
    starting: bool
    inf: float
    layer_norm: LayerNorm
    linear: Linear
    mha: Attention

    def __call__(
        self, x: Float[Array, "... I J C_in"], mask: Bool[Array, "... I J"]
    ) -> Float[Array, "... I J C_in"]:
        if not self.starting:
            x = einops.rearrange(x, "... I J C_in -> ... J I C_in")
            mask = einops.rearrange(mask, "... I J -> ... J I")

        x = self.layer_norm(x)

        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        triangle_bias = einops.rearrange(self.linear(x), "... J I H -> ... 1 H J I")
        biases = [mask_bias, triangle_bias]
        x = self.mha(q_x=x, kv_x=x, biases=biases)

        if not self.starting:
            x = einops.rearrange(x, "... J I C_in -> ... I J C_in")

        return x


@register_from_torch(boltzgen.model.layers.outer_product_mean.OuterProductMean)
class OuterProductMean(AbstractFromTorch):
    c_hidden: int
    norm: LayerNorm
    proj_a: Linear
    proj_b: Linear
    proj_o: Linear

    def __call__(
        self, m: Float[Array, "B S N D"], mask: Bool[Array, "B S N"]
    ) -> Float[Array, "B N N c_out"]:
        mask = mask[..., None]
        m = self.norm(m)
        a = self.proj_a(m) * mask
        b = self.proj_b(m) * mask

        # outer product mean
        mask = mask[:, :, None, :] * mask[:, :, :, None]
        num_mask = mask.sum(1).clip(min=1)
        z = jnp.einsum("bsic,bsjd->bijcd", a, b)
        z = einops.rearrange(z, "b i j c d -> b i j (c d)")
        z = z / num_mask
        return self.proj_o(z)


@register_from_torch(boltzgen.model.modules.transformers.AdaLN)
class AdaLN(AbstractFromTorch):
    a_norm: LayerNorm
    s_norm: LayerNorm
    s_scale: Linear
    s_bias: Linear

    def __call__(self, a, s):
        a = self.a_norm(a)
        s = self.s_norm(s)
        return jax.nn.sigmoid(self.s_scale(s)) * a + self.s_bias(s)


@register_from_torch(boltzgen.model.modules.transformers.ConditionedTransitionBlock)
class ConditionedTransitionBlock(AbstractFromTorch):
    adaln: AdaLN
    swish_gate: Sequential
    a_to_b: Linear
    b_to_a: Linear
    output_projection: Sequential

    def __call__(self, a, s):
        a = self.adaln(a, s)
        b = self.swish_gate(a) * self.a_to_b(a)
        return self.output_projection(s) * self.b_to_a(b)


def single_to_keys(single, indexing_matrix, W, H):
    B, N, D = single.shape
    K = N // W
    single = single.reshape(B, 2 * K, W // 2, D)
    r = jnp.einsum("b j i d, j k -> b k i d", single, indexing_matrix).reshape(
        B, K, H, D
    )
    return r


@register_from_torch(boltzgen.model.modules.encoders.FourierEmbedding)
class FourierEmbedding(eqx.Module):
    weight: Float[Array, "Proj Out"]
    bias: Float[Array, "Out"]

    def __call__(self, times):
        times = einops.rearrange(times, "b -> b 1")
        rand_proj = self.bias + times @ self.weight.T
        return jnp.cos(2 * jnp.pi * rand_proj)

    @staticmethod
    def from_torch(module: boltzgen.model.modules.encoders.FourierEmbedding):
        return FourierEmbedding(
            weight=from_torch(module.proj.weight), bias=from_torch(module.proj.bias)
        )


@register_from_torch(boltzgen.model.modules.encoders.SingleConditioning)
class SingleConditioning(AbstractFromTorch):
    eps: float
    sigma_data: float
    norm_single: LayerNorm
    fourier_embed: FourierEmbedding
    norm_fourier: LayerNorm
    fourier_to_single: Linear
    transitions: list[Transition]
    single_embed: Linear

    def __call__(self, times, s_trunk, s_inputs):
        s = jnp.concatenate([s_trunk, s_inputs], axis=-1)
        s = self.single_embed(self.norm_single(s))
        fourier_embed = self.fourier_embed(times)
        normed_fourier = self.norm_fourier(fourier_embed)
        fourier_to_single = self.fourier_to_single(normed_fourier)
        s = einops.rearrange(fourier_to_single, "b d -> b 1 d") + s
        for transition in self.transitions:
            s = transition(s) + s

        return s, normed_fourier


@register_from_torch(boltzgen.model.modules.encoders.PairwiseConditioning)
class PairwiseConditioning(AbstractFromTorch):
    dim_pairwise_init_proj: Sequential
    transitions: list[Transition]

    def __call__(self, z_trunk, token_rel_pos_feats):
        z = jnp.concatenate([z_trunk, token_rel_pos_feats], axis=-1)
        z = self.dim_pairwise_init_proj(z)
        for transition in self.transitions:
            z = transition(z) + z

        return z


def _copysign(a, b):
    signs_differ = (a < 0) != (b < 0)
    return jnp.where(signs_differ, -a, a)


def random_quaternions(n: int, key):
    o = jax.random.normal(key=key, shape=(n, 4))
    s = (o * o).sum(1)
    o = o / _copysign(jnp.sqrt(s), o[:, 0])[:, None]
    return o


def quaternion_to_matrix(quaternions):
    r, i, j, k = jnp.moveaxis(quaternions, -1, 0)
    two_s = 2.0 / jnp.sum(quaternions * quaternions, axis=-1)

    o = jnp.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=-1,
    )

    return o.reshape(quaternions.shape[:-1] + (3, 3))


def random_rotations(n: int, key):
    """
    Generate random rotations as 3x3 rotation matrices.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """

    return quaternion_to_matrix(random_quaternions(n, key=key))


def randomly_rotate(coords, return_second_coords=False, second_coords=None, *, key):
    R = random_rotations(len(coords), key=key)

    if return_second_coords:
        return jnp.einsum("bmd,bds->bms", coords, R), (
            jnp.einsum("bmd,bds->bms", second_coords, R)
            if second_coords is not None
            else None
        )

    return jnp.einsum("bmd,bds->bms", coords, R)


def center_random_augmentation(
    atom_coords,
    atom_mask,
    s_trans=1.0,
    augmentation=True,
    centering=True,
    return_second_coords=False,
    second_coords=None,
    *,
    key,
):
    """Center and randomly augment the input coordinates."""

    if centering:
        atom_mean = jnp.sum(
            atom_coords * atom_mask[:, :, None], axis=1, keepdims=True
        ) / jnp.sum(atom_mask[:, :, None], axis=1, keepdims=True)
        atom_coords = atom_coords - atom_mean

        if second_coords is not None:
            # apply same transformation also to this input
            second_coords = second_coords - atom_mean

    if augmentation:
        atom_coords, second_coords = randomly_rotate(
            atom_coords, return_second_coords=True, second_coords=second_coords, key=key
        )
        random_trans = (
            jax.random.normal(
                key=jax.random.fold_in(key, 1), shape=atom_coords[:, 0:1, :].shape
            )
            * s_trans
        )
        # random_trans = torch.randn_like(atom_coords[:, 0:1, :]) * s_trans
        atom_coords = atom_coords + random_trans

        if second_coords is not None:
            second_coords = second_coords + random_trans

    if return_second_coords:
        return atom_coords, second_coords

    return atom_coords


@eqx.filter_jit
def weighted_rigid_align(
    true_coords,
    pred_coords,
    weights,
    mask,
):
    batch_size, num_points, dim = true_coords.shape
    weights = (mask * weights)[..., None]

    # Compute weighted centroids
    true_centroid = (true_coords * weights).sum(axis=1, keepdims=True) / weights.sum(
        axis=1, keepdims=True
    )
    pred_centroid = (pred_coords * weights).sum(axis=1, keepdims=True) / weights.sum(
        axis=1, keepdims=True
    )

    # Center the coordinates
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    if num_points < (dim + 1):
        print(
            "Warning: The size of one of the point clouds is <= dim+1. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the weighted covariance matrix
    cov_matrix = einops.einsum(
        weights * pred_coords_centered, true_coords_centered, "b n i, b n j -> b i j"
    )

    # Compute the SVD of the covariance matrix, required float32 for svd and determinant

    U, _, Vh = jnp.linalg.svd(cov_matrix, full_matrices=False)

    # Compute the rotation matrix
    rot_matrix = jnp.einsum("b i j, b j k -> b i k", U, Vh)

    # Ensure proper rotation matrix with determinant 1
    F = jnp.tile(jnp.eye(dim)[None], (batch_size, 1, 1))
    F = F.at[:, -1, -1].set(jnp.linalg.det(rot_matrix))
    rot_matrix = vmap(lambda U, F, Vh: U @ F @ Vh)(U, F, Vh)
    rot_matrix = jax.lax.stop_gradient(rot_matrix)
    pred_centroid = jax.lax.stop_gradient(pred_centroid)

    # Apply the rotation and translation
    aligned_coords = (
        einops.einsum(true_coords_centered, rot_matrix, "b n i, b j i -> b n j")
        + pred_centroid
    )

    return aligned_coords


def compute_aggregated_metric(logits, end=1.0):
    """Compute expected value of binned metric from logits"""
    num_bins = logits.shape[-1]
    bin_width = end / num_bins
    bounds = jnp.arange(start=0.5 * bin_width, stop=end, step=bin_width)
    probs = jax.nn.softmax(logits, axis=-1)
    plddt = einops.einsum(probs, bounds, "... b, b -> ...")

    return plddt


### Boltz-2
@register_from_torch(boltzgen.model.modules.encoders.RelativePositionEncoder)
class RelativePositionEncoder(AbstractFromTorch):
    linear_layer: Linear
    r_max: int
    s_max: int
    # fix_sym_check: bool
    # cyclic_pos_enc: bool

    def __call__(self, feats: dict[str, any]):
        b_same_chain = jnp.equal(
            feats["asym_id"][:, :, None], feats["asym_id"][:, None, :]
        )
        b_same_residue = jnp.equal(
            feats["residue_index"][:, :, None], feats["residue_index"][:, None, :]
        )
        b_same_entity = jnp.equal(
            feats["entity_id"][:, :, None], feats["entity_id"][:, None, :]
        )

        d_residue = (
            feats["residue_index"][:, :, None] - feats["residue_index"][:, None, :]
        )

        if "cyclic_period" in feats:
            period = jnp.where(
                feats["cyclic_period"] > 0,
                feats["cyclic_period"],
                jnp.zeros_like(feats["cyclic_period"]) + 10000,
            )

            d_residue = jnp.where(
                jax.numpy.any(feats["cyclic_period"] > 0),
                (d_residue - period * jnp.round(d_residue / period)),
                d_residue,
            )

        d_residue = jnp.clip(
            d_residue + self.r_max,
            0,
            2 * self.r_max,
        )
        d_residue = jnp.where(
            b_same_chain, d_residue, jnp.zeros_like(d_residue) + 2 * self.r_max + 1
        )
        a_rel_pos = jax.nn.one_hot(d_residue, 2 * self.r_max + 2)

        d_token = jnp.clip(
            feats["token_index"][:, :, None]
            - feats["token_index"][:, None, :]
            + self.r_max,
            0,
            2 * self.r_max,
        )
        d_token = jnp.where(
            b_same_chain & b_same_residue,
            d_token,
            jnp.zeros_like(d_token) + 2 * self.r_max + 1,
        )
        a_rel_token = jax.nn.one_hot(d_token, 2 * self.r_max + 2)

        d_chain = jnp.clip(
            feats["sym_id"][:, :, None] - feats["sym_id"][:, None, :] + self.s_max,
            0,
            2 * self.s_max,
        )
        d_chain = jnp.where(
            (~b_same_entity),
            jnp.zeros_like(d_chain) + 2 * self.s_max + 1,
            d_chain,
        )
        # Note: added  | (~b_same_entity) based on observation of ProteinX manuscript
        a_rel_chain = jax.nn.one_hot(d_chain, 2 * self.s_max + 2)

        p = self.linear_layer(
            jnp.concatenate(
                [
                    a_rel_pos,
                    a_rel_token,
                    b_same_entity[..., None],
                    a_rel_chain,
                ],
                axis=-1,
            )
        )
        return p


@register_from_torch(boltzgen.model.layers.attention.AttentionPairBias)
class AttentionPairBias(AbstractFromTorch):
    c_s: int  # input sequence dim
    num_heads: int
    head_dim: int
    inf: float

    proj_q: Linear
    proj_k: Linear
    proj_v: Linear
    proj_g: Linear
    proj_z: Sequential
    proj_o: Linear

    def __call__(
        self,
        *,
        s: Float[Array, "B S D"],
        z: Float[Array, "B N N P"],
        mask: Bool[Array, "B N N"],
        k_in: any,
    ):
        B = s.shape[0]
        assert s.ndim == 3

        q = self.proj_q(s).reshape(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).reshape(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).reshape(B, -1, self.num_heads, self.head_dim)

        bias = self.proj_z(z)

        g = jax.nn.sigmoid(self.proj_g(s))

        attn = jnp.einsum("bihd,bjhd->bhij", q, k)
        attn = attn / (self.head_dim**0.5) + bias
        attn = attn + (1 - mask[:, None, None]) * -self.inf
        attn = jax.nn.softmax(attn, axis=-1)
        o = jnp.einsum("bhij,bjhd->bihd", attn, v)
        o = o.reshape(B, -1, self.c_s)
        return self.proj_o(g * o)


# TODO: implement dropout
@register_from_torch(boltzgen.model.layers.pairformer.PairformerLayer)
class PairformerLayer(AbstractFromTorch):
    token_z: int  # pairwise embedding dimension
    dropout: float
    num_heads: int
    attention: AttentionPairBias
    tri_mul_out: TriangleMultiplicationOutgoing
    tri_mul_in: TriangleMultiplicationIncoming
    tri_att_start: TriangleAttention
    tri_att_end: TriangleAttention
    transition_s: Transition
    transition_z: Transition
    pre_norm_s: LayerNorm
    s_post_norm: LayerNorm

    def __call__(
        self,
        s: Float[Array, "B N D"],
        z: Float[Array, "B N N P"],
        mask: Bool[Array, "B N"],
        pair_mask: Bool[Array, "B N N"],
        *,
        deterministic: bool = False,
        key,
    ):
        dropout, key = get_dropout_mask(self.dropout, z, not deterministic, key=key)
        z = z + dropout * self.tri_mul_out(z, pair_mask)

        dropout, key = get_dropout_mask(self.dropout, z, not deterministic, key=key)
        z = z + dropout * self.tri_mul_in(z, pair_mask)

        dropout, key = get_dropout_mask(self.dropout, z, not deterministic, key=key)
        z = z + dropout * self.tri_att_start(z, pair_mask)

        dropout, key = get_dropout_mask(
            self.dropout, z, not deterministic, key=key, columnwise=True
        )
        z = z + dropout * self.tri_att_end(z, pair_mask)
        z = z + self.transition_z(z)

        s_normed = self.pre_norm_s(s)
        s = s + self.attention(s=s_normed, z=z, mask=mask, k_in=s_normed)
        s = s + self.transition_s(s)
        s = self.s_post_norm(s)
        return s, z, key


@register_from_torch(boltzgen.model.layers.pairformer.PairformerModule)
class Pairformer(eqx.Module):
    stacked_parameters: PairformerLayer
    static: PairformerLayer

    @staticmethod
    def from_torch(m: boltzgen.model.layers.pairformer.PairformerModule):
        layers = [from_torch(layer) for layer in m.layers]
        _, static = eqx.partition(layers[0], eqx.is_inexact_array)
        return Pairformer(
            tree.map(
                lambda *v: jnp.stack(v, 0),
                *[eqx.filter(layer, eqx.is_inexact_array) for layer in layers],
            ),
            static,
        )

    def __call__(self, s, z, mask, pair_mask, *, key, deterministic=False):
        """
        s: Float[Array, "B N D"]
        z: Float[Array, "B N N P"]
        mask: Bool[Array, "B N"]
        pair_mask: Bool[Array, "B N N"]
        """

        @jax.checkpoint
        def _body_fn(carry, params):
            s, z, key = carry
            return eqx.combine(self.static, params)(
                s, z, mask, pair_mask, key=key, deterministic=deterministic
            ), None

        (s, z, key), _ = jax.lax.scan(
            _body_fn,
            (s, z, key),
            self.stacked_parameters,
        )

        return s, z


def get_indexing_matrix2(K, W, H):
    # Just run this in torch and np the return array...
    return np.array(boltzgen.model.modules.encoders.get_indexing_matrix(K, W, H, "cpu"))


@register_from_torch(boltzgen.model.modules.encoders.AtomEncoder)
class AtomEncoder(AbstractFromTorch):
    embed_atom_features: Linear
    embed_atompair_ref_pos: Linear
    embed_atompair_ref_dist: Linear
    embed_atompair_mask: Linear
    c_to_p_trans_k: Sequential
    c_to_p_trans_q: Sequential
    p_mlp: Sequential
    s_to_c_trans: Sequential | None
    # use_no_atom_char: bool
    # use_atom_backbone_feat: bool
    # use_residue_feats_atoms: bool
    atoms_per_window_queries: int
    atoms_per_window_keys: int
    structure_prediction: bool
    s_to_c_trans: Sequential | None
    z_to_p_trans: Sequential | None

    def __call__(
        self,
        feats: dict,
        s_trunk=None | Float[Array, "B N Ts"],
        z=None | Float[Array, "b n n tz"],
    ):
        B, N, _ = feats["ref_pos"].shape
        atom_mask = feats["atom_pad_mask"]  # Bool['b m'],

        atom_ref_pos = feats["ref_pos"]  # Float['b m 3'],
        atom_uid = feats["ref_space_uid"]  # Long['b m'],

        atom_feats = [
            atom_ref_pos,
            feats["ref_charge"][..., None],
            feats["ref_element"],
            feats["ref_atom_name_chars"].reshape(B, N, 4 * 64),
        ]

        # if self.use_residue_feats_atoms:
        #     res_feats = jnp.concatenate(
        #         [
        #             feats["res_type"],
        #             feats["modified"][..., None],
        #             jax.nn.one_hot(feats["mol_type"], num_classes=4),
        #         ],
        #         axis=-1,
        #     )
        #     atom_to_token = feats["atom_to_token"]

        #     atom_res_feats = (
        #         atom_to_token @ res_feats
        #     )  # jnp.batch#jnp.bmm(atom_to_token, res_feats)
        #     atom_feats.append(atom_res_feats)

        atom_feats = jnp.concatenate(atom_feats, axis=-1)

        c = self.embed_atom_features(atom_feats)
        # note we are already creating the windows to make it more efficient
        W, H = self.atoms_per_window_queries, self.atoms_per_window_keys
        B, N = c.shape[:2]
        K = N // W
        keys_indexing_matrix = get_indexing_matrix2(K, W, H)
        to_keys = partial(
            single_to_keys, indexing_matrix=keys_indexing_matrix, W=W, H=H
        )

        atom_ref_pos_queries = atom_ref_pos.reshape(B, K, W, 1, 3)
        atom_ref_pos_keys = to_keys(atom_ref_pos).reshape(B, K, 1, H, 3)
        d = atom_ref_pos_keys - atom_ref_pos_queries  # Float['b k w h 3']
        d_norm = jnp.sum(d * d, axis=-1, keepdims=True)  # Float['b k w h 1']
        d_norm = 1 / (1 + d_norm)  # AF3 feeds in the reciprocal of the distance norm

        atom_mask_queries = atom_mask.reshape(B, K, W, 1).astype(
            bool
        )  # Bool['b k w 1']
        atom_mask_keys = to_keys(atom_mask[..., None]).reshape(B, K, 1, H).astype(bool)
        atom_uid_queries = atom_uid.reshape(B, K, W, 1)
        atom_uid_keys = to_keys(atom_uid[..., None]).reshape(B, K, 1, H).astype(int)

        v = (
            atom_mask_queries & atom_mask_keys & (atom_uid_queries == atom_uid_keys)
        ).astype(jnp.float32)[..., None]  # Bool['b k w h 1']
        p = self.embed_atompair_ref_pos(d) * v
        p = p + self.embed_atompair_ref_dist(d_norm) * v
        p = p + self.embed_atompair_mask(v) * v

        q = c
        if self.structure_prediction:
            # run only in structure model not in initial encoding
            atom_to_token = feats["atom_to_token"]

            s_to_c = self.s_to_c_trans(s_trunk)
            s_to_c = atom_to_token @ s_to_c  # torch.bmm(atom_to_token, s_to_c)
            c = c + s_to_c

            atom_to_token_queries = atom_to_token.reshape(
                B, K, W, atom_to_token.shape[-1]
            )
            atom_to_token_keys = to_keys(atom_to_token)
            z_to_p = self.z_to_p_trans(z)
            z_to_p = jnp.einsum(
                "bijd,bwki,bwlj->bwkld",
                z_to_p,
                atom_to_token_queries,
                atom_to_token_keys,
            )
            p = p + z_to_p

        p = p + self.c_to_p_trans_q(c.reshape(B, K, W, 1, c.shape[-1]))
        p = p + self.c_to_p_trans_k(to_keys(c).reshape(B, K, 1, H, c.shape[-1]))
        p = p + self.p_mlp(p)
        return q, c, p, to_keys


@register_from_torch(boltzgen.model.modules.transformers.DiffusionTransformerLayer)
class DiffusionTransformerLayer(AbstractFromTorch):
    adaln: AdaLN
    pair_bias_attn: AttentionPairBias
    output_projection_linear: Linear
    output_projection: Sequential
    transition: ConditionedTransitionBlock
    post_lnorm: LayerNorm | Identity

    def __call__(self, *, a, s, bias, mask=None, to_keys=None):
        assert self.pair_bias_attn
        b = self.adaln(a, s)

        assert a.ndim == 3

        k_in = b
        if to_keys is not None:
            k_in = to_keys(b)
            mask = to_keys(mask[..., None])[..., 0]

        # if self.pair_bias_attn:
        b = self.pair_bias_attn(
            s=b,
            z=bias,
            mask=mask,
            k_in=k_in,
        )
        # else:
        #     b = self.no_pair_bias_attn(s=b, mask=mask, k_in=k_in)

        # TODO: precompute pair bias
        b = self.output_projection(s) * b
        a = a + b
        return self.post_lnorm(a + self.transition(a, s))


@register_from_torch(boltzgen.model.modules.transformers.DiffusionTransformer)
class DiffusionTransformer(eqx.Module):
    # pair_bias_attn: bool
    stacked_parameters: DiffusionTransformerLayer
    static: DiffusionTransformerLayer
    depth: int

    def __call__(
        self,
        a,
        s,
        bias: Float[Array, "b n n dp"] | None = None,
        mask=None | Bool[Array, "B N"],
        to_keys=None,
        multiplicity=1,
    ):
        # if self.pair_bias_attn:
        B, N, M, D = bias.shape
        L = self.depth  # len(self.layers)
        bias = bias.reshape(B, N, M, L, D // L)

        bias = einops.rearrange(bias, "... l p -> l ... p")

        @jax.checkpoint
        def body_fn(a, params_and_bias):
            # reconstitute layer
            params, bias = params_and_bias

            layer = eqx.combine(self.static, params)
            return layer(a=a, s=s, bias=bias, mask=mask, to_keys=to_keys), None

        return jax.lax.scan(body_fn, a, (self.stacked_parameters, bias))[0]

    @staticmethod
    def from_torch(m: boltzgen.model.modules.transformers.DiffusionTransformer):
        layers = [from_torch(layer) for layer in m.layers]
        _, static = eqx.partition(layers[0], eqx.is_inexact_array)
        return DiffusionTransformer(
            # ,
            tree.map(
                lambda *v: jnp.stack(v, 0),
                *[eqx.filter(layer, eqx.is_inexact_array) for layer in layers],
            ),
            static,
            depth=len(layers),
        )


@register_from_torch(boltzgen.model.modules.transformers.AtomTransformer)
class AtomTransformer(AbstractFromTorch):
    attn_window_queries: int
    attn_window_keys: int
    diffusion_transformer: DiffusionTransformer

    def __call__(self, q, c, bias, to_keys=None, mask=None, multiplicity=None):
        W = self.attn_window_queries
        H = self.attn_window_keys
        B, N, D = q.shape
        NW = N // W

        # reshape tokens
        q = q.reshape((B * NW, W, -1))
        c = c.reshape((B * NW, W, -1))
        if mask is not None:
            mask = mask.reshape(B * NW, W)
        bias = bias.reshape((bias.shape[0] * NW, W, H, -1))

        to_keys_new = lambda x: to_keys(x.reshape(B, NW * W, -1)).reshape(B * NW, H, -1)

        q = self.diffusion_transformer(
            a=q,
            s=c,
            bias=bias,
            mask=mask,
            to_keys=to_keys_new,
            multiplicity=multiplicity,
        )

        return q.reshape((B, NW * W, D))


@register_from_torch(boltzgen.model.modules.encoders.AtomAttentionEncoder)
class AtomAttentionEncoder(AbstractFromTorch):
    structure_prediction: bool
    atom_encoder: AtomTransformer
    atom_to_token_trans: Sequential
    r_to_q_trans: Sequential | None

    def __call__(
        self,
        feats: dict[str, any],
        q,
        c,
        atom_enc_bias,
        to_keys,
        r=None | Float[Array, "bm m 3"],
        multiplicity=1,
    ):
        assert multiplicity == 1, "multiplicity must be 1 for Joltz"
        B, N, _ = feats["ref_pos"].shape
        atom_mask = feats["atom_pad_mask"].astype(bool)

        if self.structure_prediction:
            r_to_q = self.r_to_q_trans(r)
            q = q + r_to_q

        q = self.atom_encoder(
            q=q,
            mask=atom_mask,
            c=c,
            bias=atom_enc_bias,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )

        q_to_a = self.atom_to_token_trans(q)
        atom_to_token = feats["atom_to_token"]
        atom_to_token = atom_to_token  # .repeat_interleave(multiplicity, 0)
        atom_to_token_mean = atom_to_token / (
            atom_to_token.sum(axis=1, keepdims=True) + 1e-6
        )
        a = (
            einops.rearrange(atom_to_token_mean, "A B C ... -> A C B ...") @ q_to_a
        )  # torch.bmm(atom_to_token_mean.transpose(1, 2), q_to_a)

        return a, q, c, to_keys


@register_from_torch(boltzgen.model.modules.trunk.InputEmbedder)
class InputEmbedder(AbstractFromTorch):
    token_s: int
    add_method_conditioning: bool
    add_modified_flag: bool
    add_cyclic_flag: bool
    add_mol_type_feat: bool

    add_binding_specification: bool
    add_ss_specification: bool
    add_design_mask_flag: bool

    atom_encoder: AtomEncoder
    atom_enc_proj_z: Sequential

    atom_attention_encoder: AtomAttentionEncoder

    res_type_encoding: Linear
    msa_profile_encoding: Linear

    method_conditioning_init: Embedding | None
    modified_conditioning_init: Embedding | None
    cyclic_conditioning_init: Embedding | None
    mol_type_conditioning_init: Embedding | None

    ss_specification_init: Embedding | None
    binding_specification_conditioning_init: Embedding | None
    design_mask_conditioning_init: Embedding | None

    def __call__(self, feats: dict[str, any], affinity: bool = False):
        res_type = feats["res_type"]
        if affinity:
            profile = feats["profile_affinity"]
            deletion_mean = feats["deletion_mean_affinity"][..., None]
        else:
            profile = feats["profile"]
            deletion_mean = feats["deletion_mean"][..., None]

        q, c, p, to_keys = self.atom_encoder(feats)
        atom_enc_bias = self.atom_enc_proj_z(p)
        a, _, _, _ = self.atom_attention_encoder(
            feats=feats,
            q=q,
            c=c,
            atom_enc_bias=atom_enc_bias,
            to_keys=to_keys,
        )

        s = (
            a
            + self.res_type_encoding(res_type)
            + self.msa_profile_encoding(
                jnp.concatenate([profile, deletion_mean], axis=-1)
            )
        )

        if self.add_method_conditioning:
            s = s + self.method_conditioning_init(feats["method_feature"])
        if self.add_modified_flag:
            s = s + self.modified_conditioning_init(feats["modified"])
        if self.add_cyclic_flag:
            cyclic = feats["cyclic_period"].clip(max=1.0)[..., None]
            s = s + self.cyclic_conditioning_init(cyclic)
        if self.add_mol_type_feat:
            s = s + self.mol_type_conditioning_init(feats["mol_type"])

        if self.add_design_mask_flag:
            s = s + self.design_mask_conditioning_init(
                feats["design_mask"].astype(jnp.int32)
            )
        if self.add_binding_specification:
            s = s + self.binding_specification_conditioning_init(feats["binding_type"])
        if self.add_ss_specification:
            s = s + self.ss_specification_init(feats["ss_type"])

        return s


@register_from_torch(boltzgen.model.layers.pairformer.PairformerNoSeqLayer)
class PairformerNoSeqLayer(AbstractFromTorch):
    token_z: int
    dropout: float

    tri_mul_out: TriangleMultiplicationOutgoing
    tri_mul_in: TriangleMultiplicationIncoming

    tri_att_start: TriangleAttention
    tri_att_end: TriangleAttention

    transition_z: Transition

    def __call__(
        self, z: Float[Array, "B N N D"], pair_mask, *, key, deterministic=False
    ):
        dropout, key = get_dropout_mask(self.dropout, z, not deterministic, key=key)
        z = z + dropout * self.tri_mul_out(z, mask=pair_mask)

        dropout, key = get_dropout_mask(self.dropout, z, not deterministic, key=key)
        z = z + dropout * self.tri_mul_in(z, mask=pair_mask)

        dropout, key = get_dropout_mask(self.dropout, z, not deterministic, key=key)
        z = z + dropout * self.tri_att_start(z, mask=pair_mask)

        dropout, key = get_dropout_mask(self.dropout, z, not deterministic, key=key)
        z = z + dropout * self.tri_att_end(z, mask=pair_mask)
        z = z + self.transition_z(z)

        return z


@register_from_torch(boltzgen.model.layers.pairformer.PairformerNoSeqModule)
class PairformerNoSeqModule(eqx.Module):
    stacked_parameters: PairformerNoSeqLayer
    static: PairformerNoSeqLayer

    def __call__(self, z, pair_mask, *, key, deterministic):
        @jax.checkpoint
        def body_fn(a, params):
            z, key = a
            # reconstitute layer
            layer = eqx.combine(self.static, params)
            return (
                layer(z, pair_mask, deterministic=deterministic, key=key),
                jax.random.fold_in(key, 0),
            ), None

        return jax.lax.scan(body_fn, (z, key), self.stacked_parameters)[0][0]

    @staticmethod
    def from_torch(m: boltzgen.model.layers.pairformer.PairformerNoSeqModule):
        layers = [from_torch(layer) for layer in m.layers]
        _, static = eqx.partition(layers[0], eqx.is_inexact_array)
        return PairformerNoSeqModule(
            tree.map(
                lambda *v: jnp.stack(v, 0),
                *[eqx.filter(layer, eqx.is_inexact_array) for layer in layers],
            ),
            static,
        )


def cdist(
    a: Float[Array, "B T N D"], b: Float[Array, "B T M D"]
) -> Float[Array, "B T N M"]:
    r = a[:, :, :, None, :] - b[:, :, None, :, :]
    return jnp.sqrt(jnp.sum(r * r, axis=-1) + 1e-8)


def cdist_no_batch(
    a: Float[Array, "T N D"], b: Float[Array, "T M D"]
) -> Float[Array, "T N M"]:
    r = a[:, :, None, :] - b[:, None, :, :]
    return jnp.sqrt(jnp.sum(r * r, axis=-1) + 1e-8)


@register_from_torch(boltzgen.model.modules.trunk.TemplateModule)
class TemplateModule(AbstractFromTorch):
    min_dist: float
    max_dist: float
    num_bins: int
    relu: any
    z_norm: LayerNorm
    v_norm: LayerNorm
    z_proj: Linear
    a_proj: Linear
    u_proj: Linear
    pairformer: PairformerNoSeqModule

    def __call__(
        self,
        z: Float[Array, "B N N D"],
        feats: dict[str],
        pair_mask: Bool[Array, "B N N"],
        *,
        key,
        deterministic,
    ):
        res_type = feats["template_restype"]
        frame_rot = feats["template_frame_rot"]
        frame_t = feats["template_frame_t"]
        frame_mask = feats["template_mask_frame"]
        cb_coords = feats["template_cb"]
        ca_coords = feats["template_ca"]
        cb_mask = feats["template_mask_cb"]
        visibility_ids = feats["visibility_ids"]
        template_mask = jnp.any(feats["template_mask"], axis=2)
        num_templates = template_mask.sum(axis=1)
        num_templates = num_templates.clip(min=1)

        # Compute pairwise masks
        b_cb_mask = cb_mask[:, :, :, None] * cb_mask[:, :, None, :]
        b_frame_mask = frame_mask[:, :, :, None] * frame_mask[:, :, None, :]

        b_cb_mask = b_cb_mask[..., None]
        b_frame_mask = b_frame_mask[..., None]

        # Compute asym mask, template features only attend within the same chain
        B, T = res_type.shape[:2]  # noqa: N806
        tmlp_pair_mask = visibility_ids[:, :, :, None] == visibility_ids[:, :, None, :]

        # Compute distogram
        cb_dists = cdist(cb_coords, cb_coords)
        boundaries = jnp.linspace(self.min_dist, self.max_dist, self.num_bins - 1)
        distogram = (cb_dists[..., None] > boundaries).sum(axis=-1).astype(jnp.int32)
        distogram = jax.nn.one_hot(distogram, num_classes=self.num_bins)
        # Compute unit vector in each frame
        frame_rot = einops.rearrange(frame_rot[:, :, None, ...], "... a b -> ... b a")
        frame_t = frame_t[:, :, None, ...][..., None]
        ca_coords = ca_coords[:, :, :, None, ...][..., None]
        vector = frame_rot @ (ca_coords - frame_t)
        norm = jnp.linalg.norm(vector, axis=-1, keepdims=True)
        unit_vector = jnp.where(norm > 0, vector / norm, jnp.zeros_like(vector))
        unit_vector = unit_vector[..., 0]

        a_tij = jnp.concatenate(
            [
                distogram,
                b_cb_mask,
                unit_vector,
                b_frame_mask,
            ],
            axis=-1,
        )
        a_tij = a_tij * tmlp_pair_mask[..., None]

        res_type_i = res_type[:, :, :, None]
        res_type_j = res_type[:, :, None, :]
        s_i = res_type_i.shape
        res_type_i = jnp.broadcast_to(
            res_type_i, (s_i[0], s_i[1], s_i[2], res_type.shape[2], s_i[4])
        )
        s_j = res_type_j.shape
        res_type_j = jnp.broadcast_to(
            res_type_j, (s_j[0], s_j[1], res_type.shape[2], s_j[3], s_j[4])
        )
        a_tij = jnp.concatenate([a_tij, res_type_i, res_type_j], axis=-1)
        a_tij = self.a_proj(a_tij)

        s_m = pair_mask.shape
        pair_mask = jnp.broadcast_to(pair_mask[:, None], (s_m[0], T, s_m[1], s_m[2]))
        pair_mask = pair_mask.reshape(B * T, *pair_mask.shape[2:])

        v = self.z_proj(self.z_norm(z[:, None])) + a_tij
        v = v.reshape(B * T, *v.shape[2:])
        v = v + self.pairformer(v, pair_mask, key=key, deterministic=deterministic)
        v = self.v_norm(v)
        v = v.reshape(B, T, *v.shape[1:])

        template_mask = template_mask[:, :, None, None, None]
        num_templates = num_templates[:, None, None, None]
        u = (v * template_mask).sum(axis=1) / num_templates

        return self.u_proj(self.relu(u))


@register_from_torch(boltzgen.model.modules.trunk.MSALayer)
class MSALayer(AbstractFromTorch):
    msa_dropout: float
    msa_transition: Transition
    pair_weighted_averaging: PairWeightedAveraging
    pairformer_layer: PairformerNoSeqLayer
    outer_product_mean: OuterProductMean

    def __call__(self, z, m, token_mask, msa_mask, *, key, deterministic):
        msa_dropout, key = get_dropout_mask(
            self.msa_dropout, m, not deterministic, key=key
        )
        m = m + msa_dropout * self.pair_weighted_averaging(m, z, token_mask)

        m = m + self.msa_transition(m)

        z = z + self.outer_product_mean(m, msa_mask)

        z = self.pairformer_layer(
            z, token_mask, key=jax.random.fold_in(key, 0), deterministic=deterministic
        )

        return z, m


@register_from_torch(boltzgen.model.modules.trunk.MSAModule)
class MSAModule(eqx.Module):
    use_paired_feature: bool
    # subsample_msa: bool
    # num_subsampled_msa: int
    msa_proj: Linear
    s_proj: Linear
    msa_dropout: float

    stacked_parameters: MSALayer
    static: MSALayer

    @staticmethod
    def from_torch(m: boltzgen.model.modules.trunk.MSAModule):
        layers = [from_torch(layer) for layer in m.layers]
        _, static = eqx.partition(layers[0], eqx.is_inexact_array)
        return MSAModule(
            m.use_paired_feature,
            # m.subsample_msa,
            # m.num_subsampled_msa,
            from_torch(m.msa_proj),
            from_torch(m.s_proj),
            m.msa_dropout,
            tree.map(
                lambda *v: jnp.stack(v, 0),
                *[eqx.filter(layer, eqx.is_inexact_array) for layer in layers],
            ),
            static,
        )

    def __call__(self, z, emb, feats, *, key, deterministic):
        """
        z: Float[Array, "B N N D"]
        emb: Float[Array, "B N P"]
        feats: dict[str, any]
        """

        msa = feats["msa"]
        # msa = jax.nn.one_hot(msa, num_classes=const.num_tokens)
        has_deletion = feats["has_deletion"][..., None]
        deletion_value = feats["deletion_value"][..., None]
        is_paired = feats["msa_paired"][..., None]
        msa_mask = feats["msa_mask"]
        token_mask = feats["token_pad_mask"].astype(jnp.float32)
        token_mask = token_mask[:, :, None] * token_mask[:, None, :]

        if self.use_paired_feature:
            m = jnp.concatenate([msa, has_deletion, deletion_value, is_paired], axis=-1)
        else:
            m = jnp.concatenate([msa, has_deletion, deletion_value], axis=-1)

        # if self.subsample_msa:
        #     msa_indices = jax.random.permutation(key, jnp.arange(msa.shape[1]))[
        #         : self.num_subsampled_msa
        #     ]
        #     m = m[:, msa_indices]
        #     msa_mask = msa_mask[:, msa_indices]

        # Compute input projections
        m = self.msa_proj(m)
        m = m + self.s_proj(emb)[:, None, ...]

        @jax.checkpoint
        def body_fn(a, params):
            z, m, key = a
            # reconstitute layer
            layer = eqx.combine(self.static, params)
            return (
                *layer(
                    z=z,
                    m=m,
                    token_mask=token_mask,
                    msa_mask=msa_mask,
                    key=key,
                    deterministic=deterministic,
                ),
                jax.random.fold_in(key, 0),
            ), None

        return jax.lax.scan(body_fn, (z, m, key), self.stacked_parameters)[0][0][
            0
        ]  # only return z


# register_from_torch(boltz.model.modules.trunkv2.FourierEmbedding)(FourierEmbedding)


@register_from_torch(boltzgen.model.modules.trunk.ContactConditioning)
class ContactConditioning(AbstractFromTorch):
    fourier_embedding: FourierEmbedding
    encoder: Linear
    encoding_unspecified: Float[Array, "tz"]
    encoding_unselected: Float[Array, "tz"]
    cutoff_min: float
    cutoff_max: float

    def __call__(self, feats):
        assert const.contact_conditioning_info["UNSPECIFIED"] == 0
        assert const.contact_conditioning_info["UNSELECTED"] == 1

        contact_conditioning = feats["contact_conditioning"][:, :, :, 2:]
        contact_threshold = feats["contact_threshold"]
        contact_threshold_normalized = (contact_threshold - self.cutoff_min) / (
            self.cutoff_max - self.cutoff_min
        )
        contact_threshold_fourier = self.fourier_embedding(
            contact_threshold_normalized.reshape(-1)
        ).reshape(contact_threshold_normalized.shape + (-1,))

        contact_conditioning = jnp.concatenate(
            [
                contact_conditioning,
                contact_threshold_normalized[..., None],
                contact_threshold_fourier,
            ],
            axis=-1,
        )
        contact_conditioning = self.encoder(contact_conditioning)

        contact_conditioning = (
            contact_conditioning
            * (
                1
                - feats["contact_conditioning"][:, :, :, 0:2].sum(
                    axis=-1, keepdims=True
                )
            )
            + self.encoding_unspecified * feats["contact_conditioning"][:, :, :, 0:1]
            + self.encoding_unselected * feats["contact_conditioning"][:, :, :, 1:2]
        )
        return contact_conditioning


@register_from_torch(boltzgen.model.modules.trunk.DistogramModule)
class DistogramModule(AbstractFromTorch):
    distogram: Linear
    num_bins: int

    def __call__(self, z: Float[Array, "B N N D"]):
        z = z + einops.rearrange(z, "b n m d -> b m n d")
        return self.distogram(z).reshape(
            z.shape[0], z.shape[1], z.shape[2], 1, self.num_bins
        )


# register_from_torch(boltz.model.modules.encodersv2.PairwiseConditioning)(PairwiseConditioning)


def stack_modules(ms):
    static = eqx.filter(ms[0], lambda x: not eqx.is_inexact_array(x))
    stacked_parameters = tree.map(
        lambda *v: jnp.stack(v, 0),
        *[eqx.filter(m, eqx.is_inexact_array) for m in ms],
    )
    return eqx.combine(stacked_parameters, static)


def eval_stacked_modules(modules, x):
    params, static = eqx.partition(modules, eqx.is_inexact_array)
    return vmap(lambda p: eqx.combine(p, static)(x), in_axes=0)(params)


@register_from_torch(
    boltzgen.model.modules.diffusion_conditioning.DiffusionConditioning
)
class DiffusionConditioning(eqx.Module):
    pairwise_conditioner: PairwiseConditioning

    atom_encoder: AtomEncoder

    atom_enc_proj_z: eqx.Module
    atom_dec_proj_z: eqx.Module
    token_trans_proj_z: eqx.Module

    def __call__(
        self,
        s_trunk: Float[Array, "B N Ts"],
        z_trunk: Float[Array, "B N N Tz"],
        relative_position_encoding: Float[Array, "B N N Tz"],
        feats: dict[str, any],
    ):
        """
        s_trunk: Float[Array, "B N Ts"]
        z_trunk: Float[Array, "B N N Tz"]
        relative_position_encoding: Float[Array, "B N N Tz"]
        feats: dict[str, any]
        """
        # Compute pairwise conditioning
        z = self.pairwise_conditioner(
            z_trunk,
            relative_position_encoding,
        )

        q, c, p, to_keys = self.atom_encoder(
            feats=feats,
            s_trunk=s_trunk,
            z=z,
        )

        def swap_and_cat(o):
            return einops.rearrange(o, "D ... P -> ... (D P)")

        atom_enc_bias = swap_and_cat(eval_stacked_modules(self.atom_enc_proj_z, p))
        atom_dec_bias = swap_and_cat(eval_stacked_modules(self.atom_dec_proj_z, p))
        token_trans_bias = swap_and_cat(
            eval_stacked_modules(self.token_trans_proj_z, z)
        )

        return q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias

    @staticmethod
    def from_torch(
        m: boltzgen.model.modules.diffusion_conditioning.DiffusionConditioning,
    ):
        return DiffusionConditioning(
            pairwise_conditioner=from_torch(m.pairwise_conditioner),
            atom_encoder=from_torch(m.atom_encoder),
            atom_enc_proj_z=stack_modules(from_torch(m.atom_enc_proj_z)),
            atom_dec_proj_z=stack_modules(from_torch(m.atom_dec_proj_z)),
            token_trans_proj_z=stack_modules(from_torch(m.token_trans_proj_z)),
        )


@register_from_torch(boltzgen.model.modules.encoders.AtomAttentionDecoder)
class AtomAttentionDecoder(AbstractFromTorch):
    a_to_q_trans: Linear
    atom_decoder: AtomTransformer
    atom_feat_to_atom_pos_update: Sequential

    def __call__(self, *, a, q, c, atom_dec_bias, feats, to_keys, multiplicity=1):
        assert multiplicity == 1
        atom_mask = feats["atom_pad_mask"]

        atom_to_token = feats["atom_to_token"]
        a_to_q = self.a_to_q_trans(a)
        a_to_q = (
            atom_to_token @ a_to_q
        )  # vmap(lambda M, v: M @ v)(atom_to_token, a_to_q)
        q = q + a_to_q
        q = self.atom_decoder(
            q=q,
            mask=atom_mask,
            c=c,
            bias=atom_dec_bias,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )
        r_update = self.atom_feat_to_atom_pos_update(q)
        return r_update


# register_from_torch(boltz.model.modules.encodersv2.SingleConditioning)(SingleConditioning)


@register_from_torch(boltzgen.model.modules.diffusion.DiffusionModule)
class DiffusionModule(AbstractFromTorch):
    single_conditioner: SingleConditioning
    atom_attention_encoder: AtomAttentionEncoder
    s_to_a_linear: Sequential
    token_transformer: DiffusionTransformer
    a_norm: LayerNorm
    atom_attention_decoder: AtomAttentionDecoder
    token_transformer_layers: list
    token_pairformer_layers: list

    def __call__(
        self,
        s_inputs: Float[Array, "b n ts"],
        s_trunk: Float[Array, "b n ts"],
        r_noisy: Float[Array, "bm m 3"],
        times: Float[Array, "bm 1 1"],
        feats,
        diffusion_conditioning,
        multiplicity=1,
        *,
        key,
    ):
        s, normed_fourier = self.single_conditioner(times, s_trunk, s_inputs)
        a, q_skip, c_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            q=diffusion_conditioning["q"],
            c=diffusion_conditioning["c"],
            atom_enc_bias=diffusion_conditioning["atom_enc_bias"],
            to_keys=diffusion_conditioning["to_keys"],
            r=r_noisy,  # Float['b m 3'],
            multiplicity=multiplicity,
        )
        # Full self-attention on token level
        a = a + self.s_to_a_linear(s)

        mask = feats["token_pad_mask"]
        a = self.token_transformer(
            a,
            mask=mask,
            s=s,
            bias=diffusion_conditioning["token_trans_bias"],
            multiplicity=multiplicity,
        )
        a = self.a_norm(a)
        r_update = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            atom_dec_bias=diffusion_conditioning["atom_dec_bias"],
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )

        return r_update


def compute_random_augmentation(s_trans=1.0, *, key):
    R = random_rotations(1, key=key)
    random_trans = (
        jax.random.normal(shape=(1, 1, 3), key=jax.random.fold_in(key, 1)) * s_trans
    )
    return R, random_trans


# def center(atom_coords, atom_mask):
#     atom_mean = torch.sum(
#         atom_coords * atom_mask[:, :, None], dim=1, keepdim=True
#     ) / torch.sum(atom_mask[:, :, None], dim=1, keepdim=True)
#     atom_coords = atom_coords - atom_mean
#     return atom_coords


def center(atom_coords, atom_mask):
    atom_mean = jnp.sum(
        atom_coords * atom_mask[:, :, None], axis=1, keepdims=True
    ) / jnp.sum(atom_mask[:, :, None], axis=1, keepdims=True)
    atom_coords = atom_coords - atom_mean
    return atom_coords


@register_from_torch(boltzgen.model.modules.diffusion.AtomDiffusion)
class AtomDiffusion(AbstractFromTorch):
    score_model: DiffusionModule
    sigma_min: float
    sigma_max: float
    sigma_data: float
    rho: float
    P_mean: float
    P_std: float
    gamma_0: float
    gamma_min: float
    noise_scale: float
    # step_scale: float
    step_scale_random: list | None
    coordinate_augmentation: bool
    coordinate_augmentation_inference: bool | None
    alignment_reverse_diff: bool
    synchronize_sigmas: bool
    time_dilation: float
    time_dilation_start: float
    time_dilation_end: float

    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / jnp.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1 / jnp.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        return jnp.log(sigma / self.sigma_data) * 0.25

    @eqx.filter_jit
    def preconditioned_network_forward(
        self,
        noised_atom_coords,  #: Float['b m 3'],
        sigma,  #: Float['b'] | Float[' '] | float,
        network_condition_kwargs: dict,
        *,
        key,
    ):
        batch = noised_atom_coords.shape[0]

        # if isinstance(sigma, float):
        sigma = jnp.full((batch,), sigma)

        padded_sigma = einops.rearrange(sigma, "b -> b 1 1")

        r_update = self.score_model(
            r_noisy=self.c_in(padded_sigma) * noised_atom_coords,
            times=self.c_noise(sigma),
            **network_condition_kwargs,
            key=key,
        )

        denoised_coords = (
            self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * r_update
        )
        return denoised_coords

    def sample_schedule(self, num_sampling_steps):
        inv_rho = 1 / self.rho

        steps = jnp.arange(num_sampling_steps, dtype=jnp.float32)
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_sampling_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data

        sigmas = jnp.pad(sigmas, (0, 1))  # last step is sigma value of 0.
        return sigmas

    def sample_schedule_dilated(self, num_sampling_steps):
        inv_rho = 1 / self.rho

        steps = np.arange(num_sampling_steps, dtype=np.float32)
        ts = steps / (num_sampling_steps - 1)

        # remap to dilate a particular interval
        def dilate(ts, start, end, dilation):
            x = end - start
            l = start
            u = 1 - end
            assert (dilation - 1) * x <= l + u, "dilation too large"

            inv_dilation = 1 / dilation
            ratio = (l + u + (1 - dilation) * x) / (l + u)
            inv_ratio = 1 / ratio
            lprime = l * ratio
            uprime = u * ratio
            xprime = x * dilation

            lower_third = ts * inv_ratio
            middle_third = (ts - lprime) * inv_dilation + l
            upper_third = (ts - (lprime + xprime)) * inv_ratio + l + x
            return (
                (ts < lprime) * lower_third
                + ((ts >= lprime) & (ts < lprime + xprime)) * middle_third
                + (ts >= lprime + xprime) * upper_third
            )

        dilated_ts = dilate(
            ts, self.time_dilation_start, self.time_dilation_end, self.time_dilation
        )
        sigmas = (
            self.sigma_max**inv_rho
            + dilated_ts * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data  # note: done by AF3 but not by EDM

        sigmas = jnp.pad(sigmas, (0, 1))  # last step is sigma value of 0.
        return sigmas

    def sample(
        self,
        atom_mask,
        num_sampling_steps,
        *,
        key,
        sample_schedule: Literal["dilated", "af3"] = "dilated",
        step_scale: float,
        noise_scale: float,
        **network_condition_kwargs,
    ):
        shape = (*atom_mask.shape, 3)



        sigmas = (
            self.sample_schedule(num_sampling_steps)
            if sample_schedule == "af3"
            else self.sample_schedule_dilated(num_sampling_steps)
        )


        gammas = jnp.where(sigmas > self.gamma_min, self.gamma_0, 0.0)


        # atom position is noise at the beginning

        @jax.checkpoint
        def sample_body_function(carry, input):
            (sigma_tm, sigma_t, gamma) = input
            atom_coords, key = carry
            random_R, random_tr = compute_random_augmentation(key=key)
            key = jax.random.fold_in(key, 1)
            atom_coords = center(atom_coords, atom_mask)
            atom_coords = jnp.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr

            t_hat = sigma_tm * (1 + gamma)
            noise_var = noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = (
                noise_scale
                * jnp.sqrt(noise_var)
                * jax.random.normal(shape=shape, key=key)
            )
            key = jax.random.fold_in(key, 1)
            atom_coords_noisy = atom_coords + eps
            atom_coords_denoised = self.preconditioned_network_forward(
                atom_coords_noisy,
                t_hat,
                network_condition_kwargs=dict(
                    **network_condition_kwargs,
                ),
                key=key,
            )

            if self.alignment_reverse_diff:
                atom_coords_noisy = weighted_rigid_align(
                    atom_coords_noisy,
                    atom_coords_denoised,
                    atom_mask,
                    atom_mask,
                )

            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            atom_coords_next = (
                atom_coords_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

            return (atom_coords_next, jax.random.fold_in(key, 0)), None

        (atom_coords, _), _ = jax.lax.scan(
            sample_body_function,
            (
                sigmas[0] * jax.random.normal(shape=shape, key=key),
                jax.random.fold_in(key, 1),
            ),
            (sigmas[:-1], sigmas[1:], gammas[1:]),
        )

        return atom_coords

    def noise_distribution(self, batch_size, *, key):
        return self.sigma_data * jnp.exp(
            self.P_mean + self.P_std * jax.random.normal(shape=(batch_size,), key=key)
        )


class ConfidenceMetrics(eqx.Module):
    pde_logits: Float[Array, "b n n 64"]
    pae_logits: Float[Array, "b n n 64"]
    plddt_logits: Float[Array, "b n 50"]
    resolved_logits: Float[Array, "b n 2"]
    plddt: Float[Array, "b n"]
    pde: Float[Array, "b n n"]
    pae: Float[Array, "b n n"]
    complex_plddt: Float[Array, "b"]
    complex_iplddt: Float[Array, "b"]
    complex_pde: Float[Array, "b"]
    complex_ipde: Float[Array, "b"]


@register_from_torch(boltzgen.model.modules.confidence.ConfidenceHeads)
class ConfidenceHeads(AbstractFromTorch):
    max_num_atoms_per_token: int
    token_level_confidence: bool
    to_pae_intra_logits: Linear
    to_pae_inter_logits: Linear
    to_pde_intra_logits: Linear
    to_pde_inter_logits: Linear
    to_plddt_logits: Linear
    to_resolved_logits: Linear

    # TODO: Rewrite all of this to use boolean manipulation instead of multiplication etc...
    def __call__(
        self,
        s: Float[Array, "b n ts"],
        z: Float[Array, "b n n tz"],
        x_pred: Float[Array, "b m 3"],
        d,
        feats: dict[str],
        pred_distogram_logits,
    ):
        asym_id_token = feats["asym_id"]
        is_same_chain = asym_id_token[..., None] == asym_id_token[..., None, :]
        is_different_chain = jnp.logical_not(is_same_chain)

        pae_intra_logits = self.to_pae_intra_logits(z)
        pae_intra_logits = pae_intra_logits * is_same_chain[..., None]

        pae_inter_logits = self.to_pae_inter_logits(z)
        pae_inter_logits = pae_inter_logits * is_different_chain[..., None]

        pae_logits = pae_inter_logits + pae_intra_logits

        pde_intra_logits = self.to_pde_intra_logits(
            z + einops.rearrange(z, "b n m tz -> b m n tz")
        )
        pde_intra_logits = pde_intra_logits * is_same_chain[..., None]

        pde_inter_logits = self.to_pde_inter_logits(
            z + einops.rearrange(z, "b n m tz -> b m n tz")
        )
        pde_inter_logits = pde_inter_logits * is_different_chain[..., None]

        pde_logits = pde_inter_logits + pde_intra_logits
        resolved_logits = self.to_resolved_logits(s)
        plddt_logits = self.to_plddt_logits(s)
        ligand_weight = 20
        non_interface_weight = 1
        interface_weight = 10

        token_type = feats["mol_type"]
        is_ligand_token = token_type == const.chain_type_ids["NONPOLYMER"]
        plddt = compute_aggregated_metric(plddt_logits)

        token_pad_mask = feats["token_pad_mask"]
        complex_plddt = (plddt * token_pad_mask).sum(axis=-1) / token_pad_mask.sum(
            axis=-1
        )

        is_contact = d < 8
        is_different_chain = (
            feats["asym_id"][..., None] != feats["asym_id"][..., None, :]
        )
        token_interface_mask = jnp.max(
            is_contact * is_different_chain * (1 - is_ligand_token)[..., None],
            axis=-1,
        )
        token_non_interface_mask = (1 - token_interface_mask) * (1 - is_ligand_token)
        iplddt_weight = (
            is_ligand_token * ligand_weight
            + token_interface_mask * interface_weight
            + token_non_interface_mask * non_interface_weight
        )
        complex_iplddt = (plddt * token_pad_mask * iplddt_weight).sum(
            axis=-1
        ) / jnp.sum(token_pad_mask * iplddt_weight, axis=-1)

        # Compute the gPDE and giPDE
        pde = compute_aggregated_metric(pde_logits, end=32)
        pred_distogram_prob = jax.nn.softmax(pred_distogram_logits, axis=-1)
        contacts = jnp.zeros((1, 1, 1, 64), dtype=pred_distogram_prob.dtype)
        contacts = contacts.at[:, :, :, :20].set(1.0)
        prob_contact = (pred_distogram_prob * contacts).sum(-1)
        token_pad_mask = feats["token_pad_mask"]
        token_pad_pair_mask = (
            token_pad_mask[..., None]
            * token_pad_mask[..., None, :]
            * (1 - jnp.eye(token_pad_mask.shape[1])[None])
        )
        token_pair_mask = token_pad_pair_mask * prob_contact
        complex_pde = (pde * token_pair_mask).sum(axis=(1, 2)) / token_pair_mask.sum(
            axis=(1, 2)
        )
        asym_id = feats["asym_id"]
        token_interface_pair_mask = token_pair_mask * (
            asym_id[..., None] != asym_id[..., None, :]
        )
        complex_ipde = (pde * token_interface_pair_mask).sum(axis=(1, 2)) / (
            token_interface_pair_mask.sum(axis=(1, 2)) + 1e-5
        )

        return ConfidenceMetrics(
            pde_logits=pde_logits,
            pae_logits=pae_logits,
            plddt_logits=plddt_logits,
            resolved_logits=resolved_logits,
            pde=pde,
            pae=compute_aggregated_metric(pae_logits, end=32),
            complex_plddt=complex_plddt,
            complex_iplddt=complex_iplddt,
            complex_pde=complex_pde,
            complex_ipde=complex_ipde,
            plddt=plddt,
        )


@register_from_torch(boltzgen.model.modules.confidence.ConfidenceModule)
class ConfidenceModule(AbstractFromTorch):
    confidence_heads: ConfidenceHeads

    max_num_atoms_per_token: int
    boundaries: Float[Array, "num_bins"]
    dist_bin_pairwise_embed: Embedding
    token_level_confidence: bool

    s_to_z: Linear
    s_to_z_transpose: Linear

    add_s_to_z_prod: bool
    s_to_z_prod_in1: Linear | None
    s_to_z_prod_in2: Linear | None
    s_to_z_prod_out: Linear | None

    s_inputs_norm: LayerNorm

    no_update_s: bool
    s_norm: LayerNorm | None
    z_norm: LayerNorm

    add_s_input_to_s: bool
    s_input_to_s: Linear | None

    add_z_input_to_z: bool
    rel_pos: RelativePositionEncoder | None
    token_bonds: Linear | None

    bond_type_feature: bool
    token_bonds_type: Embedding | None

    contact_conditioning: ContactConditioning | None

    pairformer_stack: Pairformer
    return_latent_feats: bool

    def __call__(
        self,
        s_inputs: Float[Array, "b n ts"],
        s: Float[Array, "b n ts"],
        z: Float[Array, "b n n tz"],
        x_pred: Float[Array, "b m 3"],
        feats: dict[str, any],
        pred_distogram_logits: Float[Array, "b n n 64"],
        *,
        key,
        deterministic: bool,
    ):
        s_inputs = self.s_inputs_norm(s_inputs)
        if not self.no_update_s:
            s = self.s_norm(s)

        if self.add_s_input_to_s:
            s = s + self.s_input_to_s(s_inputs)

        z = self.z_norm(z)

        if self.add_z_input_to_z:
            relative_position_encoding = self.rel_pos(feats)
            z = z + relative_position_encoding
            z = z + self.token_bonds(feats["token_bonds"])
            if self.bond_type_feature:
                z = z + self.token_bonds_type(feats["type_bonds"])
            z = z + self.contact_conditioning(feats)

        z = (
            z
            + self.s_to_z(s_inputs)[:, :, None, :]
            + self.s_to_z_transpose(s_inputs)[:, None, :, :]
        )
        if self.add_s_to_z_prod:
            z = z + self.s_to_z_prod_out(
                self.s_to_z_prod_in1(s_inputs)[:, :, None, :]
                * self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
            )

        token_to_rep_atom = feats["token_to_rep_atom"]
        if len(x_pred.shape) == 4:
            B, mult, N, _ = x_pred.shape
            x_pred = x_pred.reshape(B * mult, N, -1)
        else:
            BM, N, _ = x_pred.shape
        x_pred_repr = token_to_rep_atom @ x_pred

        def cdist(
            a: Float[Array, "B N D"], b: Float[Array, "B M D"]
        ) -> Float[Array, "B N M"]:
            r = a[:, :, None, :] - b[:, None, :, :]
            return jnp.sqrt(jnp.sum(r * r, axis=-1) + 1e-8)

        d = cdist(x_pred_repr, x_pred_repr)
        distogram = (d[..., None] > self.boundaries).sum(axis=-1).astype(jnp.int32)
        distogram = self.dist_bin_pairwise_embed(distogram)
        z = z + distogram

        mask = feats["token_pad_mask"]
        pair_mask = mask[:, :, None] * mask[:, None, :]
        s_t, z_t = self.pairformer_stack(
            s,
            z,
            mask=mask,
            pair_mask=pair_mask,
            key=key,
            deterministic=deterministic,
        )
        s = s_t
        z = z_t

        return self.confidence_heads(
            s=s,
            z=z,
            x_pred=x_pred,
            d=d,
            feats=feats,
            pred_distogram_logits=pred_distogram_logits,
        )


@register_from_torch(boltzgen.model.modules.utils.GaussianSmearing)
class GaussianSmearing(AbstractFromTorch):
    coeff: Float
    offset: Float[Array, "num_gaussians"]

    def __call__(self, dist):
        dist = dist - self.offset.reshape(1, 1, 1, -1)
        return jnp.exp(self.coeff * jnp.power(dist, 2))


@register_from_torch(boltzgen.model.modules.encoders.DistanceTokenEncoder)
class DistanceTokenEncoder(AbstractFromTorch):
    distance_gaussian_smearing: GaussianSmearing
    distance_token_bias_trans: Transition

    def __call__(
        self,
        relative_position_encoding,
        feats,
    ):
        B, N, _, _ = relative_position_encoding.shape

        token_to_bb4_atoms = feats["token_to_bb4_atoms"]
        r = feats["coords"]

        r_repr = jnp.einsum(
            "bnd,bdm->bnm",
            token_to_bb4_atoms.reshape(B, N * 4, -1),
            r.reshape(B, -1, 3),
        )

        r_repr = r_repr.reshape(B, N, 4, 3).transpose(0, 2, 1, 3)

        d = cdist(r_repr, r_repr)[..., None]
        distance_gaussian = self.distance_gaussian_smearing(d)

        relative_position_encoding = relative_position_encoding.reshape(B, 1, N, N, -1)
        relative_position_encoding = jnp.tile(
            relative_position_encoding, (1, 4, 1, 1, 1)
        )

        distance_token_bias_input = jnp.concatenate(
            (
                distance_gaussian,
                d,
                relative_position_encoding,
            ),
            axis=-1,
        )
        distance_token_bias = (
            self.distance_token_bias_trans(distance_token_bias_input)
            .transpose(0, 2, 3, 4, 1)
            .reshape(B, N, N, -1)
        )
        return distance_token_bias


@register_from_torch(boltzgen.model.modules.trunk.TokenDistanceModule)
class TokenDistanceModule(AbstractFromTorch):
    """Template module."""

    min_dist: float
    max_dist: float
    num_bins: int
    use_token_distance_feats: bool
    relu: any
    z_norm: LayerNorm
    v_norm: LayerNorm
    z_proj: Linear
    a_proj: Linear
    u_proj: Linear
    pairformer: PairformerNoSeqModule
    token_distance_encoder: DistanceTokenEncoder

    def __call__(
        self, z, feats, pair_mask, relative_position_encoding, deterministic, key
    ):
        token_distance_mask = feats["token_distance_mask"]
        token_coords = feats["center_coords"]
        dists = cdist_no_batch(token_coords, token_coords)
        boundaries = jnp.linspace(self.min_dist, self.max_dist, self.num_bins - 1)
        distogram = (dists[..., None] > boundaries).sum(axis=-1).astype(jnp.int32)
        distogram = jax.nn.one_hot(distogram, self.num_bins)
        dist_features = self.token_distance_encoder(relative_position_encoding, feats)
        a_ij = jnp.concatenate([distogram, dist_features], axis=-1)

        a_ij = a_ij * token_distance_mask[..., None]
        a_ij = self.a_proj(a_ij)
        B = a_ij.shape[0]
        v = self.z_proj(self.z_norm(z)) + a_ij
        v = v.reshape(B, *v.shape[1:])
        v = v + self.pairformer(v, pair_mask, deterministic=deterministic, key=key)
        v = self.v_norm(v)
        v = v.reshape(B, *v.shape[1:])
        u = self.u_proj(self.relu(v))
        return u


class InitialEmbedding(eqx.Module):
    s_inputs: Float[Array, "b n d"]
    s_init: Float[Array, "b n ts"]
    z_init: Float[Array, "b n n tz"]
    relative_position_encoding: Float[Array, "b n n tz"]


class TrunkState(eqx.Module):
    s: Float[Array, "b n ts"]
    z: Float[Array, "b n n tz"]


@register_from_torch(boltzgen.model.models.boltz.Boltz)
class JoltzGen(eqx.Module):
    input_embedder: InputEmbedder
    s_init: Linear
    z_init_1: Linear
    z_init_2: Linear
    rel_pos: RelativePositionEncoder
    token_bonds: Linear
    bond_type_feature: bool
    token_bonds_type: Embedding
    contact_conditioning: ContactConditioning

    s_norm: LayerNorm
    s_recycle: Linear
    z_norm: LayerNorm
    z_recycle: Linear

    distogram_module: DistogramModule
    msa_module: MSAModule
    template_module: TemplateModule
    pairformer_module: Pairformer

    diffusion_conditioning: DiffusionConditioning
    structure_module: AtomDiffusion

    token_distance_module: TokenDistanceModule

    @staticmethod
    def from_torch(m: boltzgen.model.models.boltz.Boltz):
        assert m.use_templates

        return JoltzGen(
            input_embedder=from_torch(m.input_embedder),
            s_init=from_torch(m.s_init),
            z_init_1=from_torch(m.z_init_1),
            z_init_2=from_torch(m.z_init_2),
            rel_pos=from_torch(m.rel_pos),
            token_bonds=from_torch(m.token_bonds),
            bond_type_feature=m.bond_type_feature,
            token_bonds_type=from_torch(m.token_bonds_type),
            contact_conditioning=from_torch(m.contact_conditioning),
            s_norm=from_torch(m.s_norm),
            s_recycle=from_torch(m.s_recycle),
            z_norm=from_torch(m.z_norm),
            z_recycle=from_torch(m.z_recycle),
            distogram_module=from_torch(m.distogram_module),
            template_module=from_torch(m.template_module),
            msa_module=from_torch(m.msa_module),
            pairformer_module=from_torch(m.pairformer_module),
            diffusion_conditioning=from_torch(m.diffusion_conditioning),
            structure_module=from_torch(m.structure_module),
            token_distance_module=from_torch(m.token_distance_module),
        )

    def embed_inputs(self, feats: dict[str, any]):
        s_inputs = self.input_embedder(feats)

        s_init = self.s_init(s_inputs)

        z_init = (
            self.z_init_1(s_inputs)[:, :, None] + self.z_init_2(s_inputs)[:, None, :]
        )

        relative_position_encoding = self.rel_pos(feats)

        z_init = z_init + relative_position_encoding
        z_init = z_init + self.token_bonds(feats["token_bonds"])
        if self.bond_type_feature:
            z_init = z_init + self.token_bonds_type(
                feats["type_bonds"].astype(jnp.int32)
            )
        z_init = z_init + self.contact_conditioning(feats)

        return InitialEmbedding(
            s_inputs=s_inputs,
            s_init=s_init,
            z_init=z_init,
            relative_position_encoding=relative_position_encoding,
        )

    @eqx.filter_jit
    def trunk_iteration(
        self,
        trunk_state: TrunkState,
        initial_embedding: InitialEmbedding,
        feats: dict[str, any],
        *,
        key,
        deterministic: bool,
    ):
        # Compute pairwise mask
        mask = feats["token_pad_mask"]
        pair_mask = mask[:, :, None] * mask[:, None, :]

        # Apply recycling
        s = initial_embedding.s_init + self.s_recycle(self.s_norm(trunk_state.s))
        z = initial_embedding.z_init + self.z_recycle(self.z_norm(trunk_state.z))
        #:
        z = z + self.token_distance_module(
            z,
            feats,
            pair_mask,
            initial_embedding.relative_position_encoding,
            deterministic=deterministic,
            key=key,
        )

        z = z + self.template_module(
            z, feats, pair_mask, deterministic=deterministic, key=key
        )

        z = z + self.msa_module(
            z,
            initial_embedding.s_inputs,
            feats,
            deterministic=deterministic,
            key=jax.random.fold_in(key, 0),
        )

        s, z = self.pairformer_module(
            s,
            z,
            mask=mask,
            pair_mask=pair_mask,
            deterministic=deterministic,
            key=jax.random.fold_in(key, 1),
        )
        return TrunkState(s=s, z=z), jax.random.fold_in(key, 2)

    @eqx.filter_jit
    def recycle(
        self,
        initial_embedding: InitialEmbedding,
        recycling_steps: int,
        feats: dict[str, any],
        *,
        key,
        deterministic: bool,
    ):
        trunk_state = TrunkState(
            s=jnp.zeros_like(initial_embedding.s_init),
            z=jnp.zeros_like(initial_embedding.z_init),
        )

        def body_fn(carry, _):
            trunk_state, key = carry
            trunk_state, key = self.trunk_iteration(
                trunk_state,
                initial_embedding,
                feats,
                key=key,
                deterministic=deterministic,
            )
            return (trunk_state, key), None

        state, k = jax.lax.scan(
            body_fn, (trunk_state, key), None, length=recycling_steps
        )[0]

        state = jax.lax.stop_gradient(state)

        return self.trunk_iteration(
            trunk_state=state,
            initial_embedding=initial_embedding,
            feats=feats,
            key=k,
            deterministic=deterministic,
        )

    def __call__(
        self,
        feats: dict[str],
        recycling_steps: int = 0,
        *,
        num_sampling_steps,
        deterministic,
        key,
        step_scale,
        noise_scale,
        sample_schedule="dilated"
    ):
        initial_embedding = self.embed_inputs(feats)

        trunk_state, key = self.recycle(
            initial_embedding=initial_embedding,
            recycling_steps=recycling_steps,
            feats=feats,
            key=key,
            deterministic=deterministic,
        )

        pdistogram = self.distogram_module(trunk_state.z)

        q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
            self.diffusion_conditioning(
                trunk_state.s,
                trunk_state.z,
                initial_embedding.relative_position_encoding,
                feats,
            )
        )
        with jax.default_matmul_precision("float32"):
            struct_out = self.structure_module.sample(
                s_trunk=trunk_state.s,
                s_inputs=initial_embedding.s_inputs,
                feats=feats,
                num_sampling_steps=num_sampling_steps,
                atom_mask=feats["atom_pad_mask"],
                multiplicity=1,
                diffusion_conditioning={
                    "q": q,
                    "c": c,
                    "to_keys": to_keys,
                    "atom_enc_bias": atom_enc_bias,
                    "atom_dec_bias": atom_dec_bias,
                    "token_trans_bias": token_trans_bias,
                },
                key=jax.random.fold_in(key, 2),
                step_scale=step_scale,
                noise_scale=noise_scale,
                sample_schedule=sample_schedule,
            )

        return (pdistogram, struct_out)
