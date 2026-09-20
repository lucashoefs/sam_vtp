import math
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class CameraIntrinsics:
    """A pinhole camera at the resolution its parameters were measured at."""

    __slots__ = ("fx", "fy", "cx", "cy", "width", "height")

    def __init__(
        self, fx: float, fy: float, cx: float, cy: float, width: int, height: int
    ):
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
        return CameraIntrinsics(
            self.fx, self.fy, self.cx - x0, self.cy - y0, width, height
        )

    def pad(self, left: int, top: int, width: int, height: int) -> "CameraIntrinsics":
        """Intrinsics for an image padded by (left, top) into a (width, height) canvas."""
        return CameraIntrinsics(
            self.fx, self.fy, self.cx + left, self.cy + top, width, height
        )

    def fov_deg(self) -> Tuple[float, float]:
        """(horizontal, vertical) field of view, for comparing against training."""
        return (
            2.0 * math.degrees(math.atan(0.5 * self.width / self.fx)),
            2.0 * math.degrees(math.atan(0.5 * self.height / self.fy)),
        )

    def scaled(self, h: int, w: int) -> Tuple[float, float, float, float]:
        """Intrinsics rescaled to an (h, w) grid, pixel-centre convention."""
        sx, sy = w / self.width, h / self.height
        return (
            self.fx * sx,
            self.fy * sy,
            self.cx * sx + 0.5 * sx - 0.5,
            self.cy * sy + 0.5 * sy - 0.5,
        )


# TartanGround lcam_front: 640x640, 90 deg on both axes.
NATIVE_HW = 640
NATIVE_FY = NATIVE_HW / 2.0
TARTANGROUND = CameraIntrinsics(
    NATIVE_HW / 2.0,
    NATIVE_HW / 2.0,
    NATIVE_HW / 2.0,
    NATIVE_HW / 2.0,
    NATIVE_HW,
    NATIVE_HW,
)


def downsample_depth(depth: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Reduce depth to feature resolution with nearest sampling."""
    return F.interpolate(depth, size=size, mode="nearest-exact")


# Channel scalings, chosen so each lands roughly in [-1, 1] for typical terrain.
SLOPE_STENCIL_NATIVE = 11
ROUGHNESS_WINDOW_NATIVE = 9
LOG_DEPTH_SCALE = 5.0  # log1p(150 m) ~ 5.0
SLOPE_SCALE = 90.0  # degrees
ROUGH_SCALE = 0.5  # metres of local height std
YCAM_SCALE = 5.0  # metres


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
    (1, 2),
    (4, 5),
    (7, 8),
    (0, 1),
    (3, 4),
    (6, 7),
    (1, 2),
    (4, 5),
    (7, 8),
    (0, 3),
    (5, 8),
    (4, 7),
    (3, 6),
    (1, 4),
    (2, 5),
    (4, 7),
    (4, 2),
    (6, 4),
    (4, 2),
)


def _median3(x: torch.Tensor) -> torch.Tensor:
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    h, w = x.shape[-2:]
    v = [p[..., i : i + h, j : j + w] for i in range(3) for j in range(3)]
    for a, b in _MEDIAN9_NETWORK:
        lo, hi = torch.minimum(v[a], v[b]), torch.maximum(v[a], v[b])
        v[a], v[b] = lo, hi
    return v[4]


def geometry_channels(
    depth: torch.Tensor,
    out_hw: Tuple[int, int] = (256, 256),
    min_depth: float = 0.0,
    max_depth: float = 100.0,
    cam: CameraIntrinsics = TARTANGROUND,
    depth_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Derive the label function's own geometric arguments from metric depth.

    Args:
        depth: (N, 1, H, W) metric depth in METRES. Normalised [0, 1] depth
            produces meaningless geometry, silently.
        out_hw: resolution to compute at.
        depth_valid: optional (N, 1, H, W) mask, ANDed with the range test.

    Returns:
        (N, 5, h, w): log depth, slope, roughness, height below optical centre,
        validity.
    """
    h, w = out_hw
    if depth.shape[-2:] != out_hw:
        depth = F.interpolate(depth, size=out_hw, mode="nearest-exact")
    depth = depth.float()

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
        indexing="ij",
    )
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
        gx[..., :, k:-k] = (pts[..., :, 2 * k :] - pts[..., :, : -2 * k]) / (2.0 * k)
    if h > 2 * k:
        gy[..., k:-k, :] = (pts[..., 2 * k :, :] - pts[..., : -2 * k, :]) / (2.0 * k)

    n = torch.cross(gx, gy, dim=1)
    norm = n.norm(dim=1, keepdim=True)
    ok = norm > 1e-6
    # Gravity is -y in the level camera frame, so the y component of the unit
    # normal is the cosine of the slope angle.
    cos_t = torch.where(
        ok, n[:, 1:2].abs() / norm.clamp_min(1e-12), torch.ones_like(norm)
    )
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

    return torch.cat(
        [
            torch.log1p(z.clamp_min(0.0)) / LOG_DEPTH_SCALE,
            slope / SLOPE_SCALE,
            (rough / ROUGH_SCALE).clamp(0.0, 4.0),
            (torch.where(valid, y_cam, torch.zeros_like(y_cam)) / YCAM_SCALE).clamp(
                -4.0, 4.0
            ),
            vf,
        ],
        dim=1,
    )


class GeometryEncoder(nn.Module):
    """Encode geometry channels to a backbone feature map, gated to start at zero."""

    def __init__(
        self,
        out_channels: int,
        stride: int = 8,
        width: int = 32,
        in_channels: int = 5,
        noise_std: float = 0.02,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.noise_std = float(noise_std)
        self.dropout_p = float(dropout_p)
        layers: List[nn.Module] = []
        c_in, c = in_channels, width
        n_down = max(0, int(round(math.log2(max(1, stride)))))
        for _ in range(n_down):
            layers += [
                nn.Conv2d(c_in, c, 3, stride=2, padding=1),
                nn.GroupNorm(min(8, c), c),
                nn.GELU(),
            ]
            c_in, c = c, min(c * 2, 128)
        layers.append(nn.Conv2d(c_in, out_channels, 1))
        self.net = nn.Sequential(*layers)
        self.gate = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, geo: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        if self.training:
            if self.noise_std > 0:
                geo = geo + self.noise_std * torch.randn_like(geo)
            if self.dropout_p > 0:
                keep = (
                    torch.rand(geo.shape[0], 1, 1, 1, device=geo.device)
                    >= self.dropout_p
                ).to(geo.dtype)
                geo = geo * keep
        y = self.net(geo)
        if y.shape[-2:] != tuple(out_hw):
            y = F.interpolate(y, size=out_hw, mode="bilinear", align_corners=False)
        return y * self.gate
