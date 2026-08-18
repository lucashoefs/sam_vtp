"""SAM-VTP: the complete network, in one file, depending only on PyTorch.

Why the layers live here rather than in a dependency
----------------------------------------------------
SAM-VTP is a fine-tune of SAM 2, so importing `sam2` and instantiating the
published config would be the obvious way to build it. Three reasons not to:

  * `sam2` pulls in `hydra`, `omegaconf` and `iopath` purely to turn a YAML file
    into constructor arguments that are fixed constants here. The architecture
    never varies: it is always hiera_t at d_model 256 with a 7-slot memory.
  * Those packages pin transitive versions, which constrains the usable range of
    Python and PyTorch far more than the model itself does. Nothing below needs
    anything newer than `scaled_dot_product_attention` (torch 2.0).
  * The released checkpoints were not trained against upstream `sam2`. They use
    a fixed learned prompt in place of SAM 2's interactive prompt encoder (see
    LearnedPrompt), so building the upstream class and loading with
    `strict=False` leaves the prompt randomly initialised and says nothing.

Parameter names are identical to the SAM 2 modules they came from, because the
checkpoints are keyed by those names and a rename is a silent load failure.
Read the class docstrings for what each block does.

Portions are adapted from SAM 2 (Meta Platforms, Apache 2.0); see NOTICE.

Layout
------
  1. Configuration                    SAMVTPConfig
  2. Primitives                       LayerNorm2d, MLP
  3. Position encodings               sine (2D), rotary (axial)
  4. Image encoder                    Hiera trunk + FPN neck
  5. Memory attention                 read path
  6. Memory encoder                   write path
  7. Prompt + mask decoder
  8. SAM 2 core                       the glue holding 4-7 together
  9. Geometry                         camera model, depth-derived channels
 10. SAM-VTP                          the model itself
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SAMVTPConfig",
    "SAMVTP",
    "CameraIntrinsics",
    "TARTANGROUND",
    "geometry_channels",
]


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
@dataclass
class SAMVTPConfig:
    """Every architectural constant, in one place.

    The defaults are the only combination any released SAM-VTP checkpoint was
    trained with (SAM 2.1 hiera_t). They are exposed as fields rather than
    hard-coded so that a variant can be described without editing the layers,
    but changing one makes existing checkpoints unloadable -- which the strict
    load in `SAMVTP.load_checkpoint` will tell you about rather than absorb.
    """

    # -- image encoder (Hiera-T) --------------------------------------------
    embed_dim: int = 96
    num_heads: int = 1
    stages: Tuple[int, ...] = (1, 2, 7, 2)
    global_att_blocks: Tuple[int, ...] = (5, 7, 9)
    window_pos_embed_bkg_spatial_size: Tuple[int, int] = (7, 7)
    window_spec: Tuple[int, ...] = (8, 4, 14, 7)
    q_stride: Tuple[int, int] = (2, 2)
    q_pool: int = 3
    backbone_channel_list: Tuple[int, ...] = (768, 384, 192, 96)
    fpn_top_down_levels: Tuple[int, ...] = (2, 3)
    fpn_interp_model: str = "nearest"
    scalp: int = 1

    # -- widths --------------------------------------------------------------
    d_model: int = 256          # feature width everywhere except memory
    mem_dim: int = 64           # compressed width memory is stored at

    # -- memory --------------------------------------------------------------
    num_maskmem: int = 7        # 1 conditioning frame + 6 rolling frames
    memory_attention_layers: int = 4
    dim_feedforward: int = 2048
    rope_theta: float = 10000.0
    rope_feat_sizes: Tuple[int, int] = (64, 64)
    max_obj_ptrs_in_encoder: int = 16

    # -- decoder -------------------------------------------------------------
    image_size: int = 1024      # sets the learned prompt grid: 1024/16 = 64
    backbone_stride: int = 16
    num_multimask_outputs: int = 3
    num_prompt_tokens: int = 4
    sigmoid_scale_for_mem_enc: float = 20.0
    sigmoid_bias_for_mem_enc: float = -10.0

    # -- SAM-VTP additions ---------------------------------------------------
    multimask: bool = False
    use_refine_head: bool = True
    refine_head_channels: int = 8
    use_geometry: bool = True
    geometry_hw: int = 256

    @property
    def prompt_grid(self) -> int:
        return self.image_size // self.backbone_stride

    @property
    def num_feature_levels(self) -> int:
        return 3  # high-res features are always on for this configuration


# ---------------------------------------------------------------------------
# 2. Primitives
# ---------------------------------------------------------------------------
class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of an NCHW tensor."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MLP(nn.Module):
    """Plain n-layer MLP. Named `layers` because the checkpoints are."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: type = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return torch.sigmoid(x) if self.sigmoid_output else x


def _clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


# ---------------------------------------------------------------------------
# 3. Position encodings
# ---------------------------------------------------------------------------
class PositionEmbeddingSine(nn.Module):
    """Standard 2D sine/cosine position encoding, cached per grid size.

    Note what this is *not*: it is a pure function of grid index, with no
    dependence on content. That is the property the geometry pathway in section 9
    exists to compensate for -- memory tokens carry "I am cell (i, j)", which
    means a different piece of world on every frame of a moving camera.
    """

    def __init__(self, num_pos_feats: int, temperature: float = 10000.0,
                 normalize: bool = True, scale: Optional[float] = None):
        super().__init__()
        assert num_pos_feats % 2 == 0, "expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = float(temperature)
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize must be True when scale is given")
        self.scale = 2 * math.pi if scale is None else scale
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
        y_embed = torch.arange(1, H + 1, dtype=torch.float32, device=device).view(-1, 1).repeat(1, W)
        x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=device).view(1, -1).repeat(H, 1)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        # exp(log t * k) rather than t ** k: the scalar ** tensor form builds an
        # implicit tensor that some torch/CUDA stacks refuse to capture in a graph.
        dim_t = torch.exp((2 * (dim_t // 2) / self.num_pos_feats) * math.log(self.temperature))

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        return torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1).contiguous()


def _init_t_xy(end_x: int, end_y: int, device=None):
    t = torch.arange(end_x * end_y, dtype=torch.float32, device=device)
    return (t % end_x).float(), torch.div(t, end_x, rounding_mode="floor").float()


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0, device=None):
    """Axial rotary frequencies as a complex tensor of shape (end_x*end_y, dim/2)."""
    exp = torch.arange(0, dim, 4, dtype=torch.float32, device=device)[: (dim // 4)] / dim
    freqs = 1.0 / torch.exp(exp * math.log(float(theta)))
    t_x, t_y = _init_t_xy(end_x, end_y, device=device)
    freqs_x = torch.outer(t_x, freqs)
    freqs_y = torch.outer(t_y, freqs)
    return torch.cat(
        [torch.polar(torch.ones_like(freqs_x), freqs_x),
         torch.polar(torch.ones_like(freqs_y), freqs_y)], dim=-1
    )


def apply_rotary_enc(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor,
                     repeat_freqs_k: bool = False):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = (torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
           if xk.shape[-2] != 0 else None)
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


# ---------------------------------------------------------------------------
# 4. Image encoder: Hiera trunk + FPN neck
# ---------------------------------------------------------------------------
def window_partition(x: torch.Tensor, window_size: int):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: torch.Tensor, window_size: int,
                       pad_hw: Tuple[int, int], hw: Tuple[int, int]):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(B, Hp // window_size, Wp // window_size,
                        window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    return x[:, :H, :W, :] if (Hp > H or Wp > W) else x


class PatchEmbed(nn.Module):
    def __init__(self, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3),
                 in_chans: int = 3, embed_dim: int = 96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).permute(0, 2, 3, 1)  # B C H W -> B H W C


def _do_pool(x: torch.Tensor, pool: Optional[nn.Module]) -> torch.Tensor:
    if pool is None:
        return x
    return pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class MultiScaleAttention(nn.Module):
    def __init__(self, dim: int, dim_out: int, num_heads: int, q_pool: Optional[nn.Module] = None):
        super().__init__()
        self.dim, self.dim_out, self.num_heads = dim, dim_out, num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        q, k, v = torch.unbind(qkv, 2)
        if self.q_pool is not None:
            q = _do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]
            q = q.reshape(B, H * W, self.num_heads, -1)
        x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.proj(x.transpose(1, 2).reshape(B, H, W, -1))


class MultiScaleBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, num_heads: int, mlp_ratio: float = 4.0,
                 q_stride: Optional[Tuple[int, int]] = None, window_size: int = 0):
        super().__init__()
        self.dim, self.dim_out, self.window_size = dim, dim_out, window_size
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pool, self.q_stride = None, q_stride
        if q_stride:
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False)
        self.attn = MultiScaleAttention(dim, dim_out, num_heads=num_heads, q_pool=self.pool)
        self.norm2 = nn.LayerNorm(dim_out, eps=1e-6)
        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out, num_layers=2, activation=nn.GELU)
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.dim != self.dim_out:
            shortcut = _do_pool(self.proj(x), self.pool)

        window_size = self.window_size
        pad_hw: Tuple[int, int] = (x.shape[1], x.shape[2])
        H, W = x.shape[1], x.shape[2]
        if window_size > 0:
            x, pad_hw = window_partition(x, window_size)

        x = self.attn(x)
        if self.q_stride:
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class Hiera(nn.Module):
    """Hierarchical ViT backbone. Returns one feature map per stage."""

    def __init__(self, cfg: SAMVTPConfig):
        super().__init__()
        stages, window_spec = cfg.stages, cfg.window_spec
        assert len(stages) == len(window_spec)
        depth = sum(stages)
        self.window_spec = window_spec
        self.q_stride = cfg.q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][: cfg.q_pool]
        self.global_att_blocks = cfg.global_att_blocks

        embed_dim, num_heads = cfg.embed_dim, cfg.num_heads
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.window_pos_embed_bkg_spatial_size = cfg.window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *cfg.window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, window_spec[0], window_spec[0]))

        cur_stage = 1
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dim_out = embed_dim
            window_size = window_spec[cur_stage - 1]
            if self.global_att_blocks is not None and i in self.global_att_blocks:
                window_size = 0
            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * 2)
                num_heads = int(num_heads * 2)
                cur_stage += 1
            self.blocks.append(MultiScaleBlock(
                dim=embed_dim, dim_out=dim_out, num_heads=num_heads,
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            ))
            embed_dim = dim_out

        self.channel_list = [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
        self._pos_embed_cache: Dict[Any, torch.Tensor] = {}

    def _build_pos_embed(self, h: int, w: int) -> torch.Tensor:
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + self.pos_embed_window.tile(
            [x // y for x, y in zip(pos_embed.shape, self.pos_embed_window.shape)])
        return pos_embed.permute(0, 2, 3, 1)

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        """The trunk's input position encoding, cached for inference.

        This is a bicubic upsample of a 7x7 parameter to the full stride-4 grid
        -- 128x128x96 for a 512x512 input -- and it is identical on every frame
        of a stream. Recomputing it per frame costs a bicubic interpolation and
        a tile over 1.6M elements for a result that never changes.

        Cached only when weights cannot move underneath it: training updates
        `pos_embed`, and a grad-enabled eval pass would have the cache hand back
        a tensor carrying a stale autograd graph. Both fall through to a rebuild.
        `SAMVTP.clear_inference_caches` drops it when new weights are loaded.
        """
        h, w = hw
        if self.training or torch.is_grad_enabled():
            return self._build_pos_embed(h, w)
        key = (h, w, str(self.pos_embed.device), self.pos_embed.dtype)
        cached = self._pos_embed_cache.get(key)
        if cached is None:
            cached = self._build_pos_embed(h, w)
            self._pos_embed_cache[key] = cached
        return cached

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        x = x + self._get_pos_embed(x.shape[1:3])
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.stage_ends:
                outputs.append(x.permute(0, 3, 1, 2))
        return outputs


class FpnNeck(nn.Module):
    """Feature pyramid over the trunk's stage outputs, plus their position encodings."""

    def __init__(self, cfg: SAMVTPConfig):
        super().__init__()
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=cfg.d_model)
        self.backbone_channel_list = list(cfg.backbone_channel_list)
        self.d_model = cfg.d_model
        self.fpn_interp_model = cfg.fpn_interp_model
        self.fpn_top_down_levels = list(cfg.fpn_top_down_levels)
        self.convs = nn.ModuleList()
        for dim in self.backbone_channel_list:
            conv = nn.Sequential()
            conv.add_module("conv", nn.Conv2d(dim, cfg.d_model, kernel_size=1, stride=1, padding=0))
            self.convs.append(conv)

    def forward(self, xs: List[torch.Tensor]):
        out: List[Optional[torch.Tensor]] = [None] * len(self.convs)
        pos: List[Optional[torch.Tensor]] = [None] * len(self.convs)
        prev = None
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            lateral = self.convs[n - i](xs[i])
            if i in self.fpn_top_down_levels and prev is not None:
                top_down = F.interpolate(
                    prev.to(dtype=torch.float32), scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=None if self.fpn_interp_model == "nearest" else False,
                    antialias=False,
                )
                prev = lateral + top_down
            else:
                prev = lateral
            out[i] = prev
            pos[i] = self.position_encoding(prev).to(prev.dtype)
        return out, pos


class ImageEncoder(nn.Module):
    def __init__(self, cfg: SAMVTPConfig):
        super().__init__()
        self.trunk = Hiera(cfg)
        self.neck = FpnNeck(cfg)
        self.scalp = cfg.scalp
        assert self.trunk.channel_list == self.neck.backbone_channel_list, (
            f"trunk {self.trunk.channel_list} != neck {self.neck.backbone_channel_list}")

    def forward(self, sample: torch.Tensor) -> Dict[str, Any]:
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            features, pos = features[: -self.scalp], pos[: -self.scalp]
        return {"vision_features": features[-1], "vision_pos_enc": pos, "backbone_fpn": features}


# ---------------------------------------------------------------------------
# 5. Memory attention (the read path)
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    """Multi-head attention with an optional internal down-projection."""

    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1,
                 dropout: float = 0.0, kv_in_dim: Optional[int] = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim"
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.dropout_p = dropout

    def _sep(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        return x.reshape(b, n, self.num_heads, c // self.num_heads).transpose(1, 2)

    @staticmethod
    def _merge(x: torch.Tensor) -> torch.Tensor:
        b, h, n, c = x.shape
        return x.transpose(1, 2).reshape(b, n, h * c)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q, k, v = self._sep(self.q_proj(q)), self._sep(self.k_proj(k)), self._sep(self.v_proj(v))
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
        return self.out_proj(self._merge(out))


class RoPEAttention(Attention):
    """Attention with axial rotary position encoding on q and (part of) k.

    `num_k_exclude_rope` leaves the trailing object-pointer tokens unrotated:
    they are not spatial, so giving them a spatial phase would be meaningless.
    """

    def __init__(self, *args, rope_theta: float = 10000.0, rope_k_repeat: bool = False,
                 feat_sizes: Tuple[int, int] = (64, 64), **kwargs):
        super().__init__(*args, **kwargs)
        self.rope_theta = rope_theta
        self.rope_k_repeat = rope_k_repeat
        self._head_dim = self.internal_dim // self.num_heads
        freqs = compute_axial_cis(dim=self._head_dim, end_x=int(feat_sizes[0]),
                                  end_y=int(feat_sizes[1]), theta=rope_theta)
        # Not a buffer: it is a derived constant, absent from every checkpoint,
        # and registering it would make every strict load fail on an extra key.
        self.freqs_cis = freqs

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                num_k_exclude_rope: int = 0) -> torch.Tensor:
        q, k, v = self._sep(self.q_proj(q)), self._sep(self.k_proj(k)), self._sep(self.v_proj(v))

        w = h = int(math.sqrt(q.shape[-2]))
        if self.freqs_cis.device != q.device:
            self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = compute_axial_cis(
                dim=self._head_dim, end_x=w, end_y=h, theta=self.rope_theta, device=q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat, "q/k length mismatch requires rope_k_repeat"

        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q, k[:, :, :num_k_rope], freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
        return self.out_proj(self._merge(out))


class MemoryAttentionLayer(nn.Module):
    """Self-attend the current frame, then cross-attend it against memory."""

    def __init__(self, cfg: SAMVTPConfig, dropout: float = 0.1):
        super().__init__()
        d_model = cfg.d_model
        self.d_model = d_model
        self.self_attn = RoPEAttention(
            embedding_dim=d_model, num_heads=1, downsample_rate=1, dropout=dropout,
            rope_theta=cfg.rope_theta, feat_sizes=cfg.rope_feat_sizes)
        self.cross_attn_image = RoPEAttention(
            embedding_dim=d_model, num_heads=1, downsample_rate=1, dropout=dropout,
            kv_in_dim=cfg.mem_dim, rope_theta=cfg.rope_theta,
            rope_k_repeat=True, feat_sizes=cfg.rope_feat_sizes)
        self.linear1 = nn.Linear(d_model, cfg.dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(cfg.dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        # Position encoding is added to the cross-attention keys but not to the
        # queries, and never inside self-attention: RoPE already carries position.
        self.pos_enc_at_attn = False
        self.pos_enc_at_cross_attn_queries = False
        self.pos_enc_at_cross_attn_keys = True

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                pos: torch.Tensor, query_pos: torch.Tensor,
                num_k_exclude_rope: int = 0) -> torch.Tensor:
        t2 = self.norm1(tgt)
        q = k = t2 + query_pos if self.pos_enc_at_attn else t2
        tgt = tgt + self.dropout1(self.self_attn(q, k, v=t2))

        t2 = self.norm2(tgt)
        t2 = self.cross_attn_image(
            q=t2 + query_pos if self.pos_enc_at_cross_attn_queries else t2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            num_k_exclude_rope=num_k_exclude_rope,
        )
        tgt = tgt + self.dropout2(t2)

        t2 = self.norm3(tgt)
        t2 = self.linear2(self.dropout(F.relu(self.linear1(t2))))
        return tgt + self.dropout3(t2)


class MemoryAttention(nn.Module):
    def __init__(self, cfg: SAMVTPConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.layers = _clones(MemoryAttentionLayer(cfg), cfg.memory_attention_layers)
        self.num_layers = cfg.memory_attention_layers
        self.norm = nn.LayerNorm(cfg.d_model)
        self.pos_enc_at_input = True

    def forward(self, curr: torch.Tensor, memory: torch.Tensor,
                curr_pos: torch.Tensor, memory_pos: torch.Tensor,
                num_obj_ptr_tokens: int = 0) -> torch.Tensor:
        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos
        # Layers are batch-first; the caller works in (seq, batch, channel).
        output = output.transpose(0, 1)
        curr_pos = curr_pos.transpose(0, 1)
        memory = memory.transpose(0, 1)
        memory_pos = memory_pos.transpose(0, 1)
        for layer in self.layers:
            output = layer(tgt=output, memory=memory, pos=memory_pos,
                           query_pos=curr_pos, num_k_exclude_rope=num_obj_ptr_tokens)
        return self.norm(output).transpose(0, 1)


# ---------------------------------------------------------------------------
# 6. Memory encoder (the write path)
# ---------------------------------------------------------------------------
class MaskDownSampler(nn.Module):
    """Downsample a full-resolution mask to the feature grid, widening channels."""

    def __init__(self, embed_dim: int = 256, kernel_size: int = 3, stride: int = 2,
                 padding: int = 1, total_stride: int = 16):
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride ** num_layers == total_stride
        layers: List[nn.Module] = []
        c_in = c_out = 1
        for _ in range(num_layers):
            c_out = c_in * (stride ** 2)
            layers += [nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride,
                                 padding=padding), LayerNorm2d(c_out), nn.GELU()]
            c_in = c_out
        layers.append(nn.Conv2d(c_out, embed_dim, kernel_size=1))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CXBlock(nn.Module):
    """ConvNeXt block, used to fuse image features with the encoded mask."""

    def __init__(self, dim: int, kernel_size: int = 7, padding: int = 3,
                 layer_scale_init_value: float = 1e-6, use_dwconv: bool = True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding,
                                groups=dim if use_dwconv else 1)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (nn.Parameter(layer_scale_init_value * torch.ones(dim))
                      if layer_scale_init_value > 0 else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.norm(self.dwconv(x)).permute(0, 2, 3, 1)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        if self.gamma is not None:
            x = self.gamma * x
        return inp + x.permute(0, 3, 1, 2)


class Fuser(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.proj = nn.Identity()
        self.layers = _clones(CXBlock(dim), num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class MemoryEncoder(nn.Module):
    """Compress (image features, predicted mask) into a memory slot."""

    def __init__(self, cfg: SAMVTPConfig):
        super().__init__()
        in_dim = cfg.d_model
        self.mask_downsampler = MaskDownSampler(embed_dim=in_dim)
        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = Fuser(dim=in_dim, num_layers=2)
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=cfg.mem_dim)
        self.out_proj = (nn.Conv2d(in_dim, cfg.mem_dim, kernel_size=1)
                         if cfg.mem_dim != in_dim else nn.Identity())

    def forward(self, pix_feat: torch.Tensor, masks: torch.Tensor,
                skip_mask_sigmoid: bool = False) -> Dict[str, Any]:
        if not skip_mask_sigmoid:
            masks = torch.sigmoid(masks)
        masks = self.mask_downsampler(masks)
        x = self.pix_feat_proj(pix_feat.to(masks.device)) + masks
        x = self.out_proj(self.fuser(x))
        return {"vision_features": x, "vision_pos_enc": [self.position_encoding(x).to(x.dtype)]}


# ---------------------------------------------------------------------------
# 7. Prompt + mask decoder
# ---------------------------------------------------------------------------
class LearnedPrompt(nn.Module):
    """A fixed, learned prompt replacing SAM's interactive prompt encoder.

    SAM 2 is prompted: a user clicks, and the encoder turns those clicks into
    tokens. Traversability has no user and no click -- the region of interest is
    always "the ground the robot can drive on" -- so the prompt is a constant of
    the task, and is learned once during training rather than supplied per frame.

    That makes the three tensors here the entire prompt: `sparse_tokens` stand in
    for the click embeddings, `dense_emb` for the mask-prompt channel, and
    `learned_pe` for the decoder's image position encoding. Point coordinates
    passed to `forward` are accepted and ignored, so the decoder's call site is
    unchanged.

    This is the class every released SAM-VTP checkpoint was trained with. A
    checkpoint carrying SAM's `point_embeddings` / `mask_downscaling` weights
    instead is a different model and will not load.
    """

    def __init__(self, cfg: SAMVTPConfig):
        super().__init__()
        self.embed_dim = cfg.d_model
        g = cfg.prompt_grid
        self.image_embedding_size = (g, g)
        self.input_image_size = (cfg.image_size, cfg.image_size)
        self.num_tokens = cfg.num_prompt_tokens
        self.sparse_tokens = nn.Parameter(torch.randn(cfg.num_prompt_tokens, cfg.d_model))
        self.dense_emb = nn.Parameter(torch.randn(cfg.d_model, g, g))
        self.learned_pe = nn.Parameter(torch.randn(cfg.d_model, g, g))

    def forward(self, points=None, boxes=None, masks=None):
        if points is not None:
            bs = points[0].shape[0]
        elif boxes is not None:
            bs = boxes.shape[0]
        elif masks is not None:
            bs = masks.shape[0]
        else:
            bs = 1
        return (self.sparse_tokens.unsqueeze(0).expand(bs, -1, -1),
                self.dense_emb.unsqueeze(0).expand(bs, -1, -1, -1))

    def get_dense_pe(self) -> torch.Tensor:
        return self.learned_pe.unsqueeze(0)


class TwoWayAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int = 2048,
                 attention_downsample_rate: int = 2, skip_first_layer_pe: bool = False):
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=nn.ReLU)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries, keys, query_pe, key_pe):
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            queries = queries + self.self_attn(q=q, k=q, v=queries)
        queries = self.norm1(queries)

        q, k = queries + query_pe, keys + key_pe
        queries = self.norm2(queries + self.cross_attn_token_to_image(q=q, k=k, v=keys))
        queries = self.norm3(queries + self.mlp(queries))

        q, k = queries + query_pe, keys + key_pe
        keys = self.norm4(keys + self.cross_attn_image_to_token(q=k, k=q, v=queries))
        return queries, keys


class TwoWayTransformer(nn.Module):
    def __init__(self, depth: int, embedding_dim: int, num_heads: int, mlp_dim: int,
                 attention_downsample_rate: int = 2):
        super().__init__()
        self.depth, self.embedding_dim = depth, embedding_dim
        self.num_heads, self.mlp_dim = num_heads, mlp_dim
        self.layers = nn.ModuleList([
            TwoWayAttentionBlock(embedding_dim, num_heads, mlp_dim,
                                 attention_downsample_rate, skip_first_layer_pe=(i == 0))
            for i in range(depth)
        ])
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding, image_pe, point_embedding):
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        queries, keys = point_embedding, image_embedding
        for layer in self.layers:
            queries, keys = layer(queries, keys, point_embedding, image_pe)
        q, k = queries + point_embedding, keys + image_pe
        queries = self.norm_final_attn(queries + self.final_attn_token_to_image(q=q, k=k, v=keys))
        return queries, keys


class MaskDecoder(nn.Module):
    """SAM's mask decoder: tokens attend to the image, then hypernetworks emit masks."""

    def __init__(self, cfg: SAMVTPConfig):
        super().__init__()
        d = cfg.d_model
        self.transformer_dim = d
        self.transformer = TwoWayTransformer(depth=2, embedding_dim=d, mlp_dim=2048, num_heads=8)
        self.num_multimask_outputs = cfg.num_multimask_outputs
        self.iou_token = nn.Embedding(1, d)
        self.num_mask_tokens = cfg.num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, d)
        self.pred_obj_scores = True
        self.obj_score_token = nn.Embedding(1, d)
        self.use_multimask_token_for_obj_ptr = True
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(d, d // 4, kernel_size=2, stride=2),
            LayerNorm2d(d // 4),
            nn.GELU(),
            nn.ConvTranspose2d(d // 4, d // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.use_high_res_features = True
        self.conv_s0 = nn.Conv2d(d, d // 8, kernel_size=1, stride=1)
        self.conv_s1 = nn.Conv2d(d, d // 4, kernel_size=1, stride=1)
        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(d, d, d // 8, 3) for _ in range(self.num_mask_tokens)])
        self.iou_prediction_head = MLP(d, 256, self.num_mask_tokens, 3, sigmoid_output=True)
        self.pred_obj_score_head = MLP(d, d, 1, 3)

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                dense_prompt_embeddings, multimask_output: bool, repeat_image: bool,
                high_res_features=None):
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings, image_pe, sparse_prompt_embeddings,
            dense_prompt_embeddings, repeat_image, high_res_features)

        if multimask_output:
            masks, iou_pred = masks[:, 1:, :, :], iou_pred[:, 1:]
        else:
            masks, iou_pred = masks[:, 0:1, :, :], iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]
        else:
            sam_tokens_out = mask_tokens_out[:, 0:1]
        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                      dense_prompt_embeddings, repeat_image, high_res_features=None):
        output_tokens = torch.cat(
            [self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        if repeat_image:
            # expand+reshape rather than repeat_interleave: the latter is not
            # capturable in a CUDA graph on some torch/CUDA builds.
            n_img, n_rep = image_embeddings.shape[0], int(tokens.shape[0])
            src = (image_embeddings.unsqueeze(1)
                   .expand(n_img, n_rep, *image_embeddings.shape[1:])
                   .reshape(n_img * n_rep, *image_embeddings.shape[1:]))
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings

        assert image_pe.size(0) == 1, "image_pe must have batch size 1"
        n_rep = int(tokens.shape[0])
        pos_src = (image_pe.unsqueeze(1)
                   .expand(image_pe.shape[0], n_rep, *image_pe.shape[1:])
                   .reshape(image_pe.shape[0] * n_rep, *image_pe.shape[1:]))
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 1, :]
        mask_tokens_out = hs[:, 2 : (2 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        feat_s0, feat_s1 = high_res_features
        upscaled = act1(ln1(dc1(src) + feat_s1))
        upscaled = act2(dc2(upscaled) + feat_s0)

        hyper_in = torch.stack(
            [self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
             for i in range(self.num_mask_tokens)], dim=1)
        b, c, h, w = upscaled.shape
        masks = (hyper_in @ upscaled.view(b, c, h * w)).view(b, -1, h, w)

        iou_pred = self.iou_prediction_head(iou_token_out)
        object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        return masks, iou_pred, mask_tokens_out, object_score_logits


# ---------------------------------------------------------------------------
# 8. SAM 2 core
# ---------------------------------------------------------------------------
def get_1d_sine_pe(pos_inds: torch.Tensor, dim: int, temperature: float = 10000.0):
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = torch.exp((2 * (dim_t // 2) / pe_dim) * math.log(float(temperature)))
    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    return torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)


class SAM2Core(nn.Module):
    """The SAM 2 submodules plus the logic that reads and writes memory.

    Held by `SAMVTP` under the attribute name `sam2_model`, because that is the
    prefix every released checkpoint's keys carry.

    A reduced SAM2Base. The interactive-tracking paths (click refinement, reverse
    tracking, mask-as-output, non-overlap constraints, multi-object batching,
    conditioning-frame selection) are absent, since SAM-VTP drives a single
    region forward through time under a fixed prompt and never reaches them.
    """

    def __init__(self, cfg: SAMVTPConfig):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = ImageEncoder(cfg)
        self.memory_attention = MemoryAttention(cfg)
        self.memory_encoder = MemoryEncoder(cfg)

        self.hidden_dim = cfg.d_model
        self.mem_dim = cfg.mem_dim
        self.num_maskmem = cfg.num_maskmem
        self.num_feature_levels = cfg.num_feature_levels
        self.image_size = cfg.image_size
        self.backbone_stride = cfg.backbone_stride
        self.max_obj_ptrs_in_encoder = cfg.max_obj_ptrs_in_encoder
        self.sigmoid_scale_for_mem_enc = cfg.sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = cfg.sigmoid_bias_for_mem_enc
        self.directly_add_no_mem_embed = True

        # Downsamples a mask prompt to the low-res logit stride. Unused on this
        # forward path but present in every checkpoint, so it is built to keep
        # the load strict.
        self.mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)

        self.maskmem_tpos_enc = nn.Parameter(torch.zeros(cfg.num_maskmem, 1, 1, cfg.mem_dim))
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.no_mem_pos_enc = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.no_obj_ptr = nn.Parameter(torch.zeros(1, cfg.d_model))
        self.no_obj_embed_spatial = nn.Parameter(torch.zeros(1, cfg.mem_dim))
        for p in (self.maskmem_tpos_enc, self.no_mem_embed, self.no_mem_pos_enc,
                  self.no_obj_ptr, self.no_obj_embed_spatial):
            nn.init.trunc_normal_(p, std=0.02)

        self.sam_prompt_encoder = LearnedPrompt(cfg)
        self.sam_mask_decoder = MaskDecoder(cfg)
        self.obj_ptr_proj = MLP(cfg.d_model, cfg.d_model, cfg.d_model, 3)
        self.obj_ptr_tpos_proj = nn.Linear(cfg.d_model, cfg.mem_dim)

    # -- image ---------------------------------------------------------------
    def forward_image(self, img_batch: torch.Tensor) -> Dict[str, Any]:
        out = self.image_encoder(img_batch)
        # Pre-project the two high-res levels once, as the decoder consumes them
        # on every frame.
        out["backbone_fpn"] = list(out["backbone_fpn"])
        out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(out["backbone_fpn"][0])
        out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(out["backbone_fpn"][1])
        return out

    def prepare_backbone_features(self, backbone_out: Dict[str, Any]):
        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]
        return vision_feats, vision_pos_embeds, feat_sizes

    # -- memory read ---------------------------------------------------------
    def prepare_memory_conditioned_features(
        self,
        frame_idx: int,
        current_vision_feats: List[torch.Tensor],
        current_vision_pos_embeds: List[torch.Tensor],
        feat_sizes: List[Tuple[int, int]],
        output_dict: Dict[str, Any],
    ) -> torch.Tensor:
        """Cross-attend the current frame against the memory bank.

        Memory frames are selected at unit stride: slot `t_pos` holds the frame
        `num_maskmem - t_pos` steps back. (Upstream SAM 2 supports a wider stride
        at evaluation time; this configuration never uses it, so the arithmetic
        is written out directly.)
        """
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device

        to_cat_memory: List[torch.Tensor] = []
        to_cat_memory_pos: List[torch.Tensor] = []

        cond_outputs = output_dict["cond_frame_outputs"]
        non_cond_outputs = output_dict["non_cond_frame_outputs"]
        assert len(cond_outputs) > 0, "memory conditioning needs a conditioning frame"

        # Conditioning frames all sit at t_pos = 0. There is never more than one
        # here (frame 0), so no closest-frame selection is required.
        t_pos_and_prevs = [(0, out) for out in cond_outputs.values()]
        for t_pos in range(1, self.num_maskmem):
            prev_frame_idx = frame_idx - (self.num_maskmem - t_pos)
            t_pos_and_prevs.append((t_pos, non_cond_outputs.get(prev_frame_idx)))

        for t_pos, prev in t_pos_and_prevs:
            if prev is None:
                continue
            feats = prev["maskmem_features"].to(device, non_blocking=True)
            to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
            pos = prev["maskmem_pos_enc"][-1].to(device).flatten(2).permute(2, 0, 1)
            to_cat_memory_pos.append(pos + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1])

        # Object pointers: a compact non-spatial summary of each past frame's
        # output token, carrying identity where the spatial memory carries shape.
        num_obj_ptr_tokens = 0
        # The horizon is clamped by how many frames actually exist, and the
        # temporal encoding is normalised by that same clamped horizon -- not by
        # the configured maximum. Early in a clip the two differ, and using the
        # unclamped value silently shifts every pointer's temporal phase.
        n_ptrs = min(frame_idx + 1, self.max_obj_ptrs_in_encoder)
        pos_and_ptrs = [(frame_idx - t, out["obj_ptr"])
                        for t, out in cond_outputs.items() if t <= frame_idx]
        for t_diff in range(1, n_ptrs):
            out = non_cond_outputs.get(frame_idx - t_diff)
            if out is not None:
                pos_and_ptrs.append((t_diff, out["obj_ptr"]))

        if pos_and_ptrs:
            pos_list, ptrs_list = zip(*pos_and_ptrs)
            obj_ptrs = torch.stack(ptrs_list, dim=0)
            t_diff_max = n_ptrs - 1
            obj_pos = torch.tensor(pos_list, dtype=torch.float32, device=device)
            obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=C)
            obj_pos = self.obj_ptr_tpos_proj(obj_pos)
            obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
            if self.mem_dim < C:
                # A 256-d pointer becomes four 64-d memory tokens.
                ptr_repeat = C // self.mem_dim
                obj_ptrs = obj_ptrs.reshape(-1, B, ptr_repeat, self.mem_dim)
                obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                obj_pos = (obj_pos.unsqueeze(1).expand(-1, ptr_repeat, -1, -1)
                           .reshape(-1, B, self.mem_dim))
            to_cat_memory.append(obj_ptrs)
            to_cat_memory_pos.append(obj_pos)
            num_obj_ptr_tokens = obj_ptrs.shape[0]

        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos = torch.cat(to_cat_memory_pos, dim=0)
        out = self.memory_attention(
            curr=current_vision_feats[-1], curr_pos=current_vision_pos_embeds[-1],
            memory=memory, memory_pos=memory_pos, num_obj_ptr_tokens=num_obj_ptr_tokens)
        return out.permute(1, 2, 0).view(B, C, H, W)

    # -- memory write --------------------------------------------------------
    def encode_new_memory(self, current_vision_feats, feat_sizes,
                          pred_masks_high_res, object_score_logits):
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)

        mask_for_mem = torch.sigmoid(pred_masks_high_res)
        mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        out = self.memory_encoder(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        maskmem_features = out["vision_features"]
        # Mark frames the model believes contain no traversable region, so the
        # absence is informative rather than an ambiguous all-zero memory.
        is_obj_appearing = (object_score_logits > 0).float()
        maskmem_features = maskmem_features + (
            (1 - is_obj_appearing[..., None, None])
            * self.no_obj_embed_spatial[..., None, None].expand(*maskmem_features.shape))
        return maskmem_features, out["vision_pos_enc"]


# ---------------------------------------------------------------------------
# 9. Geometry: camera model, depth-derived channels
# ---------------------------------------------------------------------------
class CameraIntrinsics:
    """A pinhole camera at the resolution its parameters were measured at.

    Every physical quantity downstream -- metres of height, degrees of slope, the
    depth-derived channels -- is recovered by unprojecting through these
    numbers. Supply the wrong ones and the output is not noisier, it is wrong by
    an amount that scales with how far the assumed field of view sits from the
    real one, while still looking entirely plausible. Read the factory
    calibration off the device; do not use a nominal FOV in deployment.
    """

    __slots__ = ("fx", "fy", "cx", "cy", "width", "height")

    def __init__(self, fx: float, fy: float, cx: float, cy: float, width: int, height: int):
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.width, self.height = int(width), int(height)

    @classmethod
    def from_fov(cls, hfov_deg: float, vfov_deg: float, width: int, height: int):
        """Nominal intrinsics from a field of view. Sanity checks only."""
        fx = 0.5 * width / math.tan(math.radians(hfov_deg) / 2.0)
        fy = 0.5 * height / math.tan(math.radians(vfov_deg) / 2.0)
        return cls(fx, fy, 0.5 * width - 0.5, 0.5 * height - 0.5, width, height)

    @classmethod
    def from_K(cls, K, width: int, height: int):
        """From a row-major 3x3 `K`, e.g. `sensor_msgs/CameraInfo.k`.

        `width` and `height` must be the resolution `K` was calibrated at, not
        the resolution you feed the model. Resizing is handled by `scaled`.
        """
        K = list(K)
        if len(K) != 9:
            raise ValueError(f"K must have 9 elements, got {len(K)}")
        return cls(fx=K[0], fy=K[4], cx=K[2], cy=K[5], width=width, height=height)

    def crop(self, x0: int, y0: int, width: int, height: int) -> "CameraIntrinsics":
        """Intrinsics for a crop starting at (x0, y0). Focal lengths are unchanged."""
        return CameraIntrinsics(self.fx, self.fy, self.cx - x0, self.cy - y0, width, height)

    def pad(self, left: int, top: int, width: int, height: int) -> "CameraIntrinsics":
        """Intrinsics for an image padded by (left, top) into a (width, height) canvas."""
        return CameraIntrinsics(self.fx, self.fy, self.cx + left, self.cy + top, width, height)

    def fov_deg(self) -> Tuple[float, float]:
        """(horizontal, vertical) field of view, for comparing against training."""
        return (2.0 * math.degrees(math.atan(0.5 * self.width / self.fx)),
                2.0 * math.degrees(math.atan(0.5 * self.height / self.fy)))

    def scaled(self, h: int, w: int) -> Tuple[float, float, float, float]:
        """Intrinsics rescaled to an (h, w) grid, pixel-centre convention.

        Uses `align_corners=False` semantics, under which a resize by factor s
        maps u -> s*(u + 0.5) - 0.5. Getting this wrong shifts the geometry by
        up to half a cell, which at feature resolution is eight native pixels.
        """
        sx, sy = w / self.width, h / self.height
        return (self.fx * sx, self.fy * sy,
                self.cx * sx + 0.5 * sx - 0.5, self.cy * sy + 0.5 * sy - 0.5)


# TartanGround lcam_front: 640x640, 90 deg on both axes.
NATIVE_HW = 640
NATIVE_FY = NATIVE_HW / 2.0
TARTANGROUND = CameraIntrinsics(NATIVE_HW / 2.0, NATIVE_HW / 2.0,
                                NATIVE_HW / 2.0, NATIVE_HW / 2.0, NATIVE_HW, NATIVE_HW)

def downsample_depth(depth: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Reduce depth to feature resolution with nearest sampling, deliberately.

    Averaging depth across a cell that straddles a discontinuity returns a
    distance at which no surface exists, placing the cell in empty space between
    foreground and background. Nearest at least always reports a real surface.
    """
    return F.interpolate(depth, size=size, mode="nearest-exact")


# Channel scalings, chosen so each lands roughly in [-1, 1] for typical terrain.
SLOPE_STENCIL_NATIVE = 11
ROUGHNESS_WINDOW_NATIVE = 9
LOG_DEPTH_SCALE = 5.0     # log1p(150 m) ~ 5.0
SLOPE_SCALE = 90.0        # degrees
ROUGH_SCALE = 0.5         # metres of local height std
YCAM_SCALE = 5.0          # metres


def _odd(n: float, lo: int = 3) -> int:
    k = max(lo, int(round(n)))
    return k if k % 2 == 1 else k + 1


# Median-of-9 as a selection network: 19 compare-exchanges, each a pair of
# elementwise min/max. Sorting nine values to take the fifth is the obvious way
# to write it, but `unfold(2,3,1).unfold(3,3,1).reshape(...)` cannot be a view,
# so it gathers nine copies of the frame into one tensor before the sort kernel
# even starts. This version reads nine shifted views and never materialises them.
#
# Bit-exact against the sort for finite input, ties included -- a correct
# selection network returns the same element, not merely an equal one. NaN is
# the exception (min/max propagate it differently than sort does), and the one
# caller passes `z_raw`, which `torch.where(valid, ...)` has already made finite.
_MEDIAN9_NETWORK = (
    (1, 2), (4, 5), (7, 8), (0, 1), (3, 4), (6, 7), (1, 2), (4, 5), (7, 8),
    (0, 3), (5, 8), (4, 7), (3, 6), (1, 4), (2, 5), (4, 7), (4, 2), (6, 4),
    (4, 2),
)


def _median3(x: torch.Tensor) -> torch.Tensor:
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    h, w = x.shape[-2:]
    v = [p[..., i:i + h, j:j + w] for i in range(3) for j in range(3)]
    for a, b in _MEDIAN9_NETWORK:
        lo, hi = torch.minimum(v[a], v[b]), torch.maximum(v[a], v[b])
        v[a], v[b] = lo, hi
    return v[4]


def geometry_channels(depth: torch.Tensor, out_hw: Tuple[int, int] = (256, 256),
                      min_depth: float = 0.0, max_depth: float = 100.0,
                      cam: CameraIntrinsics = TARTANGROUND,
                      depth_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Derive the label function's own geometric arguments from metric depth.

    The supervision is built as `semantic_prior x slope_factor x roughness_factor
    x height_gate`, where slope comes from surface normals over a wide stencil,
    roughness from the local standard deviation of height, and the gate from
    height above ground -- all deterministic functions of depth. Handing the
    network raw depth would ask it to rediscover those closed forms with
    convolutional capacity that could be doing something else; handing it these
    channels leaves only the mapping from semantics and geometry to
    traversability to be learned.

    Args:
        depth: (N, 1, H, W) metric depth in METRES. Normalised [0, 1] depth
            produces meaningless geometry, silently.
        out_hw: resolution to compute at. 256 is well above the stride-4 decoder
            and the smoothing the label applies, at a sixteenth the cost of 512.
        depth_valid: optional (N, 1, H, W) mask, ANDed with the range test.
            A stereo camera's own confidence, or a range limit tighter than
            `max_depth`, is better evidence than a threshold on the value: a
            hole-filled or spatially-filtered depth map reports plausible
            numbers where it has no measurement.

    Returns:
        (N, 5, h, w): log depth, slope, roughness, height below optical centre,
        validity.
    """
    h, w = out_hw
    if depth.shape[-2:] != out_hw:
        depth = F.interpolate(depth, size=out_hw, mode="nearest-exact")
    depth = depth.float()

    # Validity matches the label generator's own test. A wider cutoff would feed
    # the roughness window sky pixels at ~2000 m and inflate it by two orders
    # of magnitude.
    valid = torch.isfinite(depth) & (depth > min_depth) & (depth <= max_depth)
    if depth_valid is not None:
        dv = depth_valid.to(depth.device)
        if dv.shape[-2:] != out_hw:
            dv = F.interpolate(dv.float(), size=out_hw, mode="nearest-exact")
        valid = valid & dv.bool()
    z_raw = torch.where(valid, depth, torch.full_like(depth, max_depth))

    fx, fy, cx, cy = cam.scaled(h, w)
    vs, us = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing="ij")
    y_cam = (vs - cy) * z_raw / fy

    z = _median3(z_raw)
    x_cam = (us - cx) * z / fx
    y_slope = (vs - cy) * z / fy

    # Slope: normal from a wide central difference on the point cloud. The
    # stencil is scaled by focal length, not pixel count, so the baseline spans
    # the same piece of world: k pixels subtend k/fy radians, so holding k/fy
    # fixed holds the world baseline fixed. A camera with a different field of
    # view needs a different pixel stencil for the same ground patch.
    k = max(1, int(round(SLOPE_STENCIL_NATIVE * fy / NATIVE_FY)))
    pts = torch.cat([x_cam, y_slope, z], dim=1)
    gx = torch.zeros_like(pts)
    gy = torch.zeros_like(pts)
    if w > 2 * k:
        gx[..., :, k:-k] = (pts[..., :, 2 * k:] - pts[..., :, : -2 * k]) / (2.0 * k)
    if h > 2 * k:
        gy[..., k:-k, :] = (pts[..., 2 * k:, :] - pts[..., : -2 * k, :]) / (2.0 * k)

    n = torch.cross(gx, gy, dim=1)
    norm = n.norm(dim=1, keepdim=True)
    ok = norm > 1e-6
    # Gravity is -y in the level camera frame, so the y component of the unit
    # normal is the cosine of the slope angle.
    cos_t = torch.where(ok, n[:, 1:2].abs() / norm.clamp_min(1e-12), torch.ones_like(norm))
    slope = torch.rad2deg(torch.acos(cos_t.clamp(0.0, 1.0)))
    slope = torch.where(ok & valid, slope, torch.zeros_like(slope))

    win = _odd(ROUGHNESS_WINDOW_NATIVE * fy / NATIVE_FY)
    vf = valid.to(depth.dtype)
    hgt = torch.where(valid, y_cam, torch.zeros_like(y_cam))
    ones = torch.ones(1, 1, win, win, device=depth.device, dtype=depth.dtype)
    pad = win // 2
    s1 = F.conv2d(F.pad(hgt * vf, (pad,) * 4), ones)
    s2 = F.conv2d(F.pad(hgt * hgt * vf, (pad,) * 4), ones)
    cnt = F.conv2d(F.pad(vf, (pad,) * 4), ones)
    cs = cnt.clamp_min(1.0)
    var = (s2 / cs - (s1 / cs) ** 2).clamp_min(0.0)
    rough = torch.where(cnt >= 4, var.sqrt(), torch.zeros_like(var))

    return torch.cat([
        torch.log1p(z.clamp_min(0.0)) / LOG_DEPTH_SCALE,
        slope / SLOPE_SCALE,
        (rough / ROUGH_SCALE).clamp(0.0, 4.0),
        (torch.where(valid, y_cam, torch.zeros_like(y_cam)) / YCAM_SCALE).clamp(-4.0, 4.0),
        vf,
    ], dim=1)


class GeometryEncoder(nn.Module):
    """Encode geometry channels to a backbone feature map, gated to start at zero.

    The output gate is zero-initialised, so a model with geometry enabled is
    bit-identical to one without it at step 0 and can only move away by learning.
    That makes the with/without-geometry ablation exact rather than approximate.

    `noise_std` and `dropout_p` apply in training only: TartanGround depth is
    exact, stereo on a real platform is not, and a model that has only ever seen
    perfect geometry leans on it harder than it should.
    """

    def __init__(self, out_channels: int, stride: int = 8, width: int = 32,
                 in_channels: int = 5, noise_std: float = 0.02, dropout_p: float = 0.1):
        super().__init__()
        self.noise_std = float(noise_std)
        self.dropout_p = float(dropout_p)
        layers: List[nn.Module] = []
        c_in, c = in_channels, width
        n_down = max(0, int(round(math.log2(max(1, stride)))))
        for _ in range(n_down):
            layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1),
                       nn.GroupNorm(min(8, c), c), nn.GELU()]
            c_in, c = c, min(c * 2, 128)
        layers.append(nn.Conv2d(c_in, out_channels, 1))
        self.net = nn.Sequential(*layers)
        self.gate = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, geo: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        if self.training:
            if self.noise_std > 0:
                geo = geo + self.noise_std * torch.randn_like(geo)
            if self.dropout_p > 0:
                keep = (torch.rand(geo.shape[0], 1, 1, 1, device=geo.device)
                        >= self.dropout_p).to(geo.dtype)
                geo = geo * keep
        y = self.net(geo)
        if y.shape[-2:] != tuple(out_hw):
            y = F.interpolate(y, size=out_hw, mode="bilinear", align_corners=False)
        return y * self.gate


# ---------------------------------------------------------------------------
# 10. SAM-VTP
# ---------------------------------------------------------------------------
class _FusedGroupNorm(nn.GroupNorm):
    """GroupNorm computed as one reduction and one affine, for `num_groups=1`.

    With a single group the statistics are per-SAMPLE scalars over (C, H, W),
    so the whole layer is `x * a + b` for per-channel `a`, `b` derived from two
    numbers. `F.group_norm` does not exploit that: measured on an Orin Nano at
    512x512 with C=8 it takes 3.44 ms in NCHW and 4.22 ms in NHWC -- it gets
    *slower* in the layout the surrounding convolutions want, because one group
    means the reduction spans every channel, contiguous in NCHW and strided in
    NHWC. This form is layout-agnostic and measured 0.593 ms.

    Algebraically the same operation: 4.8e-07 against `F.group_norm` in fp32,
    which is fp32 rounding. Subclassing rather than replacing keeps `weight` and
    `bias` where they are, so existing checkpoints load unchanged.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_groups != 1:
            return super().forward(x)
        dims = tuple(range(1, x.dim()))
        shape = (1, -1) + (1,) * (x.dim() - 2)
        # Reductions accumulate in fp32 even for a half input, so a bank of
        # 2.1M values does not lose the mean to rounding.
        var, mean = torch.var_mean(x, dim=dims, keepdim=True, unbiased=False)
        scale = self.weight.view(shape) * torch.rsqrt(var + self.eps)
        return torch.addcmul(self.bias.view(shape) - mean * scale, x, scale)


class RefinementHead(nn.Module):
    """Full-resolution RGB-guided residual refinement of the coarse mask.

    The decoder emits masks at stride 4 of the backbone embedding -- 128x128 for
    a 512x512 input -- then upsamples 16x in area, so detail thinner than about
    four input pixels (cables, fence posts, kerbs) is unrepresentable. The only
    source of sub-128 detail is the RGB image, so this head predicts a residual
    on the upsampled logits conditioned on [RGB, coarse_logits] at full
    resolution.

    Tiny by design (~1.5k parameters at 8 channels) to fit an edge budget, and
    zero-initialised at the output, so it is an exact identity at load time.

    Cheap in arithmetic and expensive in wall clock if you let it be: 0.46 GFLOP,
    but 4 and 8 channels at full resolution are the worst case for NCHW
    convolution kernels. On an Orin Nano the two 3x3 convolutions took 9.9 ms in
    NCHW and 1.6 ms in NHWC, so the head holds itself in `channels_last` and
    converts its input once. With `_FusedGroupNorm` that is 14.1 ms -> 2.6 ms
    for the same output to within one fp16 ulp.
    """

    def __init__(self, channels: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(4, channels, kernel_size=3, padding=1),
            _FusedGroupNorm(1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.out = nn.Conv2d(channels, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.to(memory_format=torch.channels_last)

    def forward(self, coarse_logits: torch.Tensor, image_norm: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image_norm, coarse_logits], dim=1)
        # The model hands this head NCHW, so the conversion is real work, but it
        # measured 0.06 ms against the 7.5 ms the convolutions save.
        x = x.contiguous(memory_format=torch.channels_last)
        # `out` has one channel, where NHWC and NCHW are the same bytes, so the
        # residual add needs no conversion back.
        return coarse_logits + self.out(self.stem(x))


def warp_state(state: Dict[str, torch.Tensor]) -> str:
    """Classify a checkpoint's relationship to the removed ego-motion warp.

    Returns "absent", "inert", or "trained".

    Presence of `no_corr_embed` alone does NOT mean a checkpoint used the warp.
    The trainer allocated that parameter unconditionally, so every no-warp
    checkpoint carries it too -- at exactly its zero initialisation, because
    nothing ever read it. Rejecting on the key would therefore refuse the very
    checkpoints this build is for. The trained ones are distinguishable: theirs
    has moved off zero (0.048 absmax on the seed-42 warp run, 0.0 on both
    geometry-only runs).

    An inert embed is safe to drop: with `warp_memory=False` the aligned-memory
    path was never entered, so the weights are numerically identical to a model
    that never had the parameter.
    """
    t = state.get("no_corr_embed")
    if t is None:
        return "absent"
    return "trained" if float(t.abs().max()) > 0.0 else "inert"


class SAMVTP(nn.Module):
    """Video traversability prediction with a geometry-aligned temporal memory.

    Per frame:
      1. the image encoder produces multi-scale features;
      2. optionally, depth-derived geometry is added to the coarsest level;
      3. memory attention conditions those features on the memory bank;
      4. the mask decoder emits traversability logits under a fixed learned prompt;
      5. a residual head recovers sub-stride detail from full-resolution RGB;
      6. the memory encoder writes the frame back into the bank.

    Input images are RGB. Set `input_normalization_mode` to match your source
    (`"0_255"`, `"0_1"`, `"-1_1"`, or `"auto"` to guess per batch); `"auto"`
    guesses from the observed range and can be fooled by a dark frame, so pin it
    explicitly in deployment.

    Depth must be metric planar z-depth in METRES, at full input resolution.
    Poses are camera-to-world `[x y z qx qy qz qw]` in a NED world frame.
    """

    def __init__(self, cfg: Optional[SAMVTPConfig] = None, **overrides):
        super().__init__()
        self.cfg = cfg = cfg if cfg is not None else SAMVTPConfig(**overrides)

        self.num_maskmem = cfg.num_maskmem
        self.multimask_output = cfg.multimask
        self.use_geometry = cfg.use_geometry
        self.geometry_hw = cfg.geometry_hw
        self.camera = TARTANGROUND
        self.input_normalization_mode = "0_255"
        self._max_non_cond_frames = max(0, cfg.num_maskmem - 1)

        # The bank holds a fixed number of *frames*, so at a high frame rate it
        # spans very little motion: at 35 FPS a 7-frame bank covers ~0.2 s
        # against ~1.2 s in training. Writing every k-th frame restores a
        # training-like window and cuts memory-encoder calls proportionally.
        # Predictions are still emitted for every frame.
        self.memory_write_every = 1
        self._frames_seen = 0

        # Runtime component switches, as distinct from `cfg`. The config says
        # which weights EXIST; these say which of them are used on this run. A
        # switch can only ever turn a component off or back on -- it cannot
        # conjure weights the checkpoint does not carry, and set_components()
        # refuses rather than running a randomly-initialised module.
        self.use_memory = True
        self.use_refine_head = cfg.use_refine_head
        self._trained_components: Dict[str, bool] = {
            "memory": True,                    # any SAM-VTP checkpoint has memory
            "geometry": bool(cfg.use_geometry),
            "refine": bool(cfg.use_refine_head),
        }
        # Bumped on every switch change so a captured CUDA graph -- which bakes
        # the configuration into its control flow -- can detect that it is stale
        # instead of silently replaying the old one.
        self._component_epoch = 0
        # Constants derived from weights and grid size; see `_prompt_tensors`.
        self._prompt_cache: Dict[Any, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.sam2_model = SAM2Core(cfg)

        self.refine_head = RefinementHead(cfg.refine_head_channels) if cfg.use_refine_head else None

        self.geometry_encoder = None
        if cfg.use_geometry:
            self.geometry_encoder = GeometryEncoder(
                out_channels=cfg.d_model, stride=max(1, cfg.geometry_hw // 32))

        self.reset_memory()

    # -- state ---------------------------------------------------------------
    def reset_memory(self) -> None:
        """Drop all temporal state. Call between independent sequences."""
        self.frame_idx = 0
        self._frames_seen = 0
        self.output_dict: Dict[str, "OrderedDict[int, Dict[str, Any]]"] = {
            "cond_frame_outputs": OrderedDict(),
            "non_cond_frame_outputs": OrderedDict(),
        }

    @property
    def needs_depth(self) -> bool:
        """True when depth is required for the model to behave as trained.

        Only the geometry channels consume it, and they are a function of depth
        and intrinsics alone -- log depth, slope, roughness, height, validity.
        """
        return bool(self.use_geometry)

    @property
    def needs_pose(self) -> bool:
        """Always False. Retained so existing callers keep working.

        Nothing in this model reads a camera pose. The ego-motion warp that once
        did was removed; it cost a localisation dependency and did not measurably
        improve accuracy. Callers may stop looking up transforms entirely.
        """
        return False

    @property
    def needs_geometry(self) -> bool:
        """Deprecated alias for `needs_depth`; prefer the two properties above.

        Kept because callers outside this repo test it before supplying inputs.
        """
        return self.needs_depth

    # -- component switches --------------------------------------------------
    #
    # Three independent parts can be turned off at runtime:
    #
    #   memory    temporal memory bank. Off = frame-wise: every frame is
    #             predicted from scratch, nothing is written or read.
    #   geometry  the depth-derived feature added to the coarsest backbone level.
    #   refine    the residual refinement head on the upsampled logits.
    #
    # Switching a component OFF on a checkpoint trained with it is a deliberate
    # ablation, not a neutral configuration change: the remaining weights were
    # fitted alongside it and are now running outside their training
    # distribution. That is a legitimate measurement -- it is how the RGB-only
    # and frame-wise rows of the paper's ablation are produced -- but it is not
    # the same model as one TRAINED without the component, and the two should
    # never be reported as if they were. `component_deviations` exists so a log
    # line can tell them apart, and `describe()` prints it.
    COMPONENTS: Tuple[str, ...] = ("memory", "geometry", "refine")

    @property
    def component_epoch(self) -> int:
        """Increments on every switch change. Used to invalidate captured graphs."""
        return self._component_epoch

    @property
    def active_components(self) -> Dict[str, bool]:
        """Which components are in use right now."""
        return {
            "memory": bool(self.use_memory),
            "geometry": bool(self.use_geometry),
            "refine": bool(self.use_refine_head and self.refine_head is not None),
        }

    @property
    def trained_components(self) -> Dict[str, bool]:
        """Which components the loaded weights were trained with."""
        return dict(self._trained_components)

    @property
    def component_deviations(self) -> Dict[str, bool]:
        """Active components that differ from the as-trained configuration.

        Empty when the model is running as trained. Non-empty means the numbers
        this produces belong to an ablation, and saying so is the difference
        between a result and a mistake.
        """
        active, trained = self.active_components, self._trained_components
        return {k: v for k, v in active.items() if trained.get(k) != v}

    def set_components(
        self,
        *,
        memory: Optional[bool] = None,
        geometry: Optional[bool] = None,
        refine: Optional[bool] = None,
    ) -> "SAMVTP":
        """Turn components on or off. `None` leaves a component unchanged.

        Turning one ON is only possible when the checkpoint actually carries its
        weights; otherwise this raises rather than running a module at random
        initialisation, which would look like a working model producing quietly
        meaningless output.

        Call before the first frame. Changing switches mid-sequence leaves a
        memory bank written under the old configuration, so this resets memory
        whenever `memory` changes.

        Returns self, so it chains: `build_model(...).set_components(memory=False)`.
        """
        requested = {"memory": memory, "geometry": geometry, "refine": refine}
        requested = {k: v for k, v in requested.items() if v is not None}
        if not requested:
            return self

        missing = {
            "geometry": self.geometry_encoder is None,
            "refine": self.refine_head is None,
        }
        for name, want in requested.items():
            if want and missing.get(name, False):
                raise ValueError(
                    f"cannot enable '{name}': this checkpoint carries no weights for it "
                    f"(trained with {self._trained_components}). Load a checkpoint that "
                    "was trained with it instead -- enabling it here would run a "
                    "randomly-initialised module.")

        before = self.active_components
        if "memory" in requested:
            self.use_memory = bool(requested["memory"])
        if "geometry" in requested:
            self.use_geometry = bool(requested["geometry"])
        if "refine" in requested:
            self.use_refine_head = bool(requested["refine"])

        if self.active_components != before:
            self._component_epoch += 1
        if before["memory"] != self.use_memory:
            self.reset_memory()
        return self

    @contextmanager
    def components(self, **switches: Optional[bool]) -> Iterator["SAMVTP"]:
        """Apply switches for a block, then restore them.

            with model.components(geometry=False):
                rgb_only = model.forward_frame(rgb, is_init_frame=True)

        Memory is reset on entry and on exit whenever the temporal path changes,
        so the block cannot inherit or leak a bank written under the other
        configuration.
        """
        previous = self.active_components
        try:
            yield self.set_components(**switches)
        finally:
            self.set_components(**previous)

    @property
    def memory_fill_level(self) -> float:
        """Fraction of the memory bank that holds real frames, in [0, 1].

        Predictions before the bank is full are made from less temporal context
        than the model was trained with. A consumer can gate planning on this
        rather than trusting the first frames of a stream.
        """
        n = (len(self.output_dict["cond_frame_outputs"])
             + len(self.output_dict["non_cond_frame_outputs"]))
        return min(1.0, n / max(1, self.num_maskmem))

    @torch.no_grad()
    def warmup(self, n: int = 3, hw: Tuple[int, int] = (512, 512)) -> None:
        """Run synthetic frames to pay lazy initialisation cost off the clock.

        The first inference allocates cuDNN workspaces, chooses convolution
        algorithms and fills the position-encoding and rotary caches, which
        together cost far more than a steady-state frame. Call this after
        loading and before the first real frame, or the robot pays it live.

        Resets memory afterwards, so the bank does not start full of noise.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        image = torch.zeros(1, 3, int(hw[0]), int(hw[1]), device=device, dtype=dtype)
        depth = None
        if self.needs_depth:
            depth = torch.full((1, 1, int(hw[0]), int(hw[1])), 5.0,
                               device=device, dtype=dtype)
        was_training = self.training
        self.eval()
        for i in range(max(1, int(n))):
            self.forward_frame(image, is_init_frame=(i == 0), depth=depth)
        self.reset_memory()
        if was_training:
            self.train()

    def normalize_imagenet(self, image: torch.Tensor) -> torch.Tensor:
        mode = self.input_normalization_mode
        if mode == "0_255":
            image = image / 255.0
        elif mode == "0_1":
            pass
        elif mode == "-1_1":
            image = (image + 1.0) / 2.0
        elif mode == "auto":
            if image.max() > 1.0:
                image = image / 255.0
            elif image.min() < 0.0:
                image = (image + 1.0) / 2.0
        else:
            raise ValueError(f"unknown input_normalization_mode {mode!r}")
        return (image.clamp(0.0, 1.0) - self.pixel_mean) / self.pixel_std

    # -- prompt --------------------------------------------------------------
    def _prompt_tensors(self, pix_feat: torch.Tensor, B: int):
        """The learned prompt and decoder position encoding, at the feature grid.

        The prompt is a constant of the task rather than a per-frame input (see
        `LearnedPrompt`), so all three tensors are fixed once the feature grid
        size is fixed. Both are stored at the 64x64 grid implied by
        `image_size`, and a 512x512 input decodes at 32x32, so the unconditional
        path resamples 2 x 256x64x64 on every frame to produce a constant.

        Shared by the eager path and the CUDA-graph runner so the two cannot
        drift apart. Caching follows the same rule as `Hiera._get_pos_embed`:
        never while training or with grad live, and dropped by
        `clear_inference_caches` when weights change.
        """
        size = tuple(pix_feat.shape[-2:])

        def build():
            sparse, dense = self.sam2_model.sam_prompt_encoder(
                points=None, boxes=None, masks=None)
            if dense.shape[-2:] != size:
                dense = F.interpolate(dense, size=size, mode="bilinear", align_corners=False)
            pe = self.sam2_model.sam_prompt_encoder.get_dense_pe()
            if pe.shape[-2:] != size:
                pe = F.interpolate(pe, size=size, mode="bilinear", align_corners=False)
            return sparse, dense, pe.to(device=pix_feat.device, dtype=pix_feat.dtype)

        if self.training or torch.is_grad_enabled():
            sparse, dense, pe = build()
        else:
            # Keyed on the parameter's device/dtype as well as the feature
            # grid's: under autocast the two differ, and a `.half()` on the
            # model would otherwise leave a stale fp32 entry reachable.
            prompt_param = self.sam2_model.sam_prompt_encoder.dense_emb
            key = (size, str(pix_feat.device), pix_feat.dtype,
                   str(prompt_param.device), prompt_param.dtype)
            cached = self._prompt_cache.get(key)
            if cached is None:
                cached = build()
                self._prompt_cache[key] = cached
            sparse, dense, pe = cached
        return sparse.expand(B, -1, -1), dense.expand(B, -1, -1, -1), pe

    def clear_inference_caches(self) -> None:
        """Drop every cached tensor derived from weights.

        The inference caches hold constants computed from parameters, so they
        are only valid while those parameters hold still. Anything that can move
        them -- loading a checkpoint, switching components -- must call this, or
        the model keeps serving a constant derived from weights it no longer has.
        """
        self.sam2_model.image_encoder.trunk._pos_embed_cache.clear()
        self._prompt_cache.clear()

    # -- forward -------------------------------------------------------------
    def forward_frame(
        self,
        image: torch.Tensor,
        is_init_frame: bool = False,
        update_memory: bool = True,
        depth: Optional[torch.Tensor] = None,
        depth_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict one frame and, unless gated, write it into memory.

        Args:
            image: (B, 3, H, W) RGB, in the range named by
                `input_normalization_mode`.
            is_init_frame: treat this frame as the conditioning frame.
            update_memory: emit a prediction without writing to the bank.
            depth: (B, 1, H, W) metric planar z-depth in metres.
            depth_valid: optional (B, 1, H, W) mask. Non-positive depths are
                already treated as missing, so this is for evidence a threshold
                cannot express: the camera's own confidence, a range limit, or
                a depth map whose holes have been filled with plausible values.

        Raises:
            ValueError: if the model was trained with geometry but depth is
                missing. Running without it is not a graceful degradation -- it
                silently disables a pathway the weights were fitted with.
        """
        if self.needs_depth and depth is None:
            raise ValueError(
                "this checkpoint was trained with use_geometry, so it needs depth "
                "in METRES per frame. Pass it, or call "
                "model.set_components(geometry=False) to run the RGB-only "
                "ablation deliberately.")

        B, _, H, W = image.shape
        image_norm = self.normalize_imagenet(image)

        backbone_out = self.sam2_model.forward_image(image_norm)

        if self.use_geometry and depth is not None:
            # Injected into the coarsest level, which becomes vision_feats[-1]
            # and so reaches both memory conditioning and the decoder from a
            # single insertion point.
            top = backbone_out["backbone_fpn"][-1]
            geo = geometry_channels(depth, out_hw=(self.geometry_hw, self.geometry_hw),
                                    cam=self.camera, depth_valid=depth_valid)
            backbone_out["backbone_fpn"] = list(backbone_out["backbone_fpn"])
            backbone_out["backbone_fpn"][-1] = top + self.geometry_encoder(
                geo.to(top.dtype), top.shape[-2:])

        vision_feats, vision_pos_embeds, feat_sizes = \
            self.sam2_model.prepare_backbone_features(backbone_out)

        has_memory = bool(self.output_dict["cond_frame_outputs"]
                          or self.output_dict["non_cond_frame_outputs"])
        # `use_memory=False` takes the same path as a first frame, every frame:
        # the bank is neither read nor written, which is the frame-wise ablation.
        if (not self.use_memory) or (not has_memory) or is_init_frame or self.frame_idx == 0:
            # The "no memory yet" token goes back into vision_feats, not just
            # into pix_feat, so it also reaches the memory encoder below. Upstream
            # SAM 2 keeps the two separate; the conditioning frame's stored memory
            # here depends on the addition, so changing it moves every
            # checkpoint's output from frame 1 onward.
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
            pix_feat = vision_feats[-1].permute(1, 2, 0).view(B, -1, *feat_sizes[-1])
        else:
            pix_feat = self.sam2_model.prepare_memory_conditioned_features(
                frame_idx=self.frame_idx,
                current_vision_feats=[vision_feats[-1]],
                current_vision_pos_embeds=[vision_pos_embeds[-1]],
                feat_sizes=[feat_sizes[-1]],
                output_dict=self.output_dict,
            )

        sparse_embeddings, dense_embeddings, image_pe = self._prompt_tensors(pix_feat, B)

        high_res_feats = [f.permute(1, 2, 0).view(B, -1, *s)
                          for f, s in zip(vision_feats[:-1], feat_sizes[:-1])]

        low_res_masks, iou_preds, obj_ptr, _ = self.sam2_model.sam_mask_decoder(
            image_embeddings=pix_feat,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
            repeat_image=False,
            high_res_features=high_res_feats,
        )

        if self.multimask_output:
            best = iou_preds.argmax(dim=1)
            rows = torch.arange(B, device=image.device)
            mask_logits = low_res_masks[rows, best].unsqueeze(1)
            obj_ptr = obj_ptr.gather(1, best.view(B, 1, 1).expand(-1, 1, obj_ptr.shape[-1]))
        else:
            mask_logits = low_res_masks
        obj_ptr = obj_ptr.squeeze(1)

        scale = max(H, W) / max(mask_logits.shape[-2], mask_logits.shape[-1])
        mask_logits = F.interpolate(
            mask_logits, size=(H, W),
            mode="bicubic" if scale >= 3 else "bilinear", align_corners=False)

        if self.refine_head is not None and self.use_refine_head:
            mask_logits = self.refine_head(mask_logits, image_norm)

        if not self.use_memory:
            update_memory = False
        if update_memory and self.memory_write_every > 1:
            if (self._frames_seen % self.memory_write_every) != 0:
                update_memory = False
        self._frames_seen += 1

        if not update_memory:
            return {"logits": mask_logits}

        object_score_logits = torch.any(
            mask_logits.flatten(1) > 0.0, dim=1, keepdim=True).float()
        maskmem_features, maskmem_pos_enc = self.sam2_model.encode_new_memory(
            current_vision_feats=vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=mask_logits,
            object_score_logits=object_score_logits,
        )

        entry: Dict[str, Any] = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "obj_ptr": obj_ptr,
        }
        if self.frame_idx == 0:
            self.output_dict["cond_frame_outputs"][self.frame_idx] = entry
        else:
            self.output_dict["non_cond_frame_outputs"][self.frame_idx] = entry
            if len(self.output_dict["non_cond_frame_outputs"]) > self._max_non_cond_frames:
                self.output_dict["non_cond_frame_outputs"].popitem(last=False)

        self.frame_idx += 1
        return {"logits": mask_logits}

    def forward(
        self,
        image_sequence: torch.Tensor,
        depth_sequence: Optional[torch.Tensor] = None,
        reset_memory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run a whole clip.

        Args:
            image_sequence: (B, T, 3, H, W) RGB.
            depth_sequence: (B, T, 1, H, W) metric depth in metres.
        """
        if reset_memory:
            self.reset_memory()
        logits = []
        for t in range(image_sequence.shape[1]):
            out = self.forward_frame(
                image_sequence[:, t],
                is_init_frame=(t == 0 and reset_memory),
                depth=None if depth_sequence is None else depth_sequence[:, t],
            )
            logits.append(out["logits"])
        return {"logits": torch.stack(logits, dim=1)}

    # -- checkpoints ---------------------------------------------------------
    @staticmethod
    def config_from_state_dict(state: Dict[str, torch.Tensor], **overrides) -> SAMVTPConfig:
        """Infer the architecture from a checkpoint's own tensors.

        The checkpoint is the authority, not the caller's flags: a mismatch here
        is not a warning but a model that quietly differs from the one whose
        numbers were published.
        """
        if warp_state(state) == "trained":
            raise RuntimeError(
                "this checkpoint was trained with the ego-motion memory warp: it "
                "carries a NON-ZERO 'no_corr_embed'. This build no longer "
                "implements the warp -- it did not measurably improve accuracy "
                "and required a camera pose, i.e. a localisation dependency, at "
                "inference. Loading it here would silently drop the aligned-memory "
                "pathway the weights were fitted with and score a model nobody "
                "trained. Use a checkpoint trained without the warp, or check out "
                "a revision that still supports it.")
        has = lambda p: any(k.startswith(p) for k in state)  # noqa: E731
        cfg_kwargs: Dict[str, Any] = dict(
            use_refine_head=has("refine_head."),
            use_geometry=has("geometry_encoder."),
        )
        if cfg_kwargs["use_refine_head"]:
            cfg_kwargs["refine_head_channels"] = int(state["refine_head.out.weight"].shape[1])
        if "sam2_model.maskmem_tpos_enc" in state:
            cfg_kwargs["num_maskmem"] = int(state["sam2_model.maskmem_tpos_enc"].shape[0])
        if "sam2_model.sam_prompt_encoder.dense_emb" in state:
            grid = int(state["sam2_model.sam_prompt_encoder.dense_emb"].shape[-1])
            cfg_kwargs["image_size"] = grid * SAMVTPConfig.backbone_stride
            cfg_kwargs["num_prompt_tokens"] = int(
                state["sam2_model.sam_prompt_encoder.sparse_tokens"].shape[0])
        cfg_kwargs.update(overrides)
        return SAMVTPConfig(**cfg_kwargs)

    def load_checkpoint(self, state: Dict[str, torch.Tensor]) -> None:
        """Load weights strictly, reporting exactly what did not match.

        Not `strict=False`: the geometry encoder and the refinement head are
        both top-level modules, so a lenient load drops every one of their
        tensors without reporting it and runs an ablated model under the full
        model's name.
        """
        if warp_state(state) == "inert":
            # Allocated unconditionally by the trainer and never read, so it is
            # still at zero. Dropping it is exact, not lenient -- but say so,
            # because a silent key drop is how the geometry encoder went missing
            # from a whole ablation table once already.
            state = {k: v for k, v in state.items() if k != "no_corr_embed"}
            print("[sam-vtp] dropped an unused zero 'no_corr_embed' (the removed "
                  "ego-motion warp); the remaining weights are unaffected.")

        missing, unexpected = self.load_state_dict(state, strict=False)
        # New weights invalidate every constant derived from the old ones.
        self.clear_inference_caches()
        if missing or unexpected:
            lines = ["checkpoint does not match this architecture."]
            if missing:
                lines.append(f"  {len(missing)} missing: {sorted(missing)[:8]}")
            if unexpected:
                lines.append(f"  {len(unexpected)} unexpected: {sorted(unexpected)[:8]}")
            if any("point_embeddings" in k or "mask_downscaling" in k for k in unexpected):
                lines.append("  This looks like a SAM 2 interactive prompt encoder; "
                             "SAM-VTP uses a fixed learned prompt (see LearnedPrompt).")
            raise RuntimeError("\n".join(lines))
