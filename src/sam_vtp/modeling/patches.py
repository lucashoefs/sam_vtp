from functools import partial
import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.sam.transformer import RoPEAttention


class CachedHiera(Hiera):
    """Hiera with the position embedding memoised on (h, w). Numerics unchanged."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_embed_cache: Dict[Any, torch.Tensor] = {}

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        if torch.is_grad_enabled():
            # pos_embed is a Parameter, so the result carries a grad_fn.
            # Caching it would pin the autograd graph for the process lifetime.
            return super()._get_pos_embed(hw)
        key = (int(hw[0]), int(hw[1]), str(self.pos_embed.device))
        cached = self._pos_embed_cache.get(key)
        if cached is None:
            cached = super()._get_pos_embed(hw)
            self._pos_embed_cache[key] = cached
        return cached


class CachedPositionEmbeddingSine(PositionEmbeddingSine):
    """Sine PE with SAM-VTP's cache, expand instead of repeat, and exp(k log t)."""

    def __init__(self, *args, **kwargs):
        # Upstream defaults warmup_cache=True, which allocates on CUDA at
        # construction time for 1024/stride grids we never run, filling a
        # cache this subclass does not read.
        kwargs.setdefault("warmup_cache", False)
        super().__init__(*args, **kwargs)
        self._cache: Dict[Any, torch.Tensor] = {}

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W = x.shape[0], x.shape[-2], x.shape[-1]
        key = (str(x.device), H, W)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._build(x.device, H, W)
            self._cache[key] = cached
        return cached[None].expand(B, -1, -1, -1)

    def _build(self, device: torch.device, H: int, W: int) -> torch.Tensor:
        y_embed = (
            torch.arange(1, H + 1, dtype=torch.float32, device=device)
            .view(-1, 1)
            .repeat(1, W)
        )
        x_embed = (
            torch.arange(1, W + 1, dtype=torch.float32, device=device)
            .view(1, -1)
            .repeat(H, 1)
        )
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        # exp(log t * k) rather than t ** k: the scalar ** tensor form builds an
        # implicit tensor that some torch/CUDA stacks refuse to capture in a graph.
        dim_t = torch.exp(
            (2 * (dim_t // 2) / self.num_pos_feats) * math.log(self.temperature)
        )

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        return torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1).contiguous()


def _init_t_xy(end_x: int, end_y: int, device=None):
    t = torch.arange(end_x * end_y, dtype=torch.float32, device=device)
    return (t % end_x).float(), torch.div(t, end_x, rounding_mode="floor").float()


def compute_axial_cis(
    dim: int, end_x: int, end_y: int, theta: float = 10000.0, device=None
):
    """Axial rotary frequencies as a complex tensor of shape (end_x*end_y, dim/2)."""
    exp = (
        torch.arange(0, dim, 4, dtype=torch.float32, device=device)[: (dim // 4)] / dim
    )
    freqs = 1.0 / torch.exp(exp * math.log(float(theta)))
    t_x, t_y = _init_t_xy(end_x, end_y, device=device)
    freqs_x = torch.outer(t_x, freqs)
    freqs_y = torch.outer(t_y, freqs)
    return torch.cat(
        [
            torch.polar(torch.ones_like(freqs_x), freqs_x),
            torch.polar(torch.ones_like(freqs_y), freqs_y),
        ],
        dim=-1,
    )


def apply_rotary_enc_vtp(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0
        else None
    )
    shape = [d if i >= xq_.ndim - 2 else 1 for i, d in enumerate(xq_.shape)]
    freqs_cis = freqs_cis.view(*shape)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        return xq_out.type_as(xq), xk
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        freqs_cis = freqs_cis.unsqueeze(2).expand(-1, -1, r, -1, -1).flatten(2, 3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RoPEAttentionVTP(RoPEAttention):
    """RoPEAttention with SAM-VTP's capture-safe frequency computation.

    Same trap as PositionEmbeddingSine and get_1d_sine_pe: upstream writes
    theta ** k, SAM 2 writes exp(k * log theta). No parameters,
    so key parity is unaffected.
    """

    def __init__(self, *args, rope_theta=10000.0, feat_sizes=(64, 64), **kwargs):
        super().__init__(*args, rope_theta=rope_theta, feat_sizes=feat_sizes, **kwargs)
        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        freqs = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = freqs.to("cuda") if torch.cuda.is_available() else freqs

    def forward(self, q, k, v, num_k_exclude_rope: int = 0):
        q = self._separate_heads(self.q_proj(q), self.num_heads)
        k = self._separate_heads(self.k_proj(k), self.num_heads)
        v = self._separate_heads(self.v_proj(v), self.num_heads)

        w = h = int(math.sqrt(q.shape[-2]))
        if self.freqs_cis.device != q.device:
            self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            # device=q.device is load-bearing. SAM 2 builds this
            # table on the GPU; polar() and exp() differ from their CPU
            # versions by ~1 ulp, so computing on host and transferring gives
            # the right numbers to 6 places and the wrong last bit.
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h, device=q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        # apply_rotary_enc_vtp has no is_cuda branch: upstream uses .repeat() on
        # CUDA and expand+flatten on CPU. Measured inert here, kept for exactness.
        q, k[:, :, :num_k_rope] = apply_rotary_enc_vtp(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0
        )
        return self.out_proj(self._recombine_heads(out))
