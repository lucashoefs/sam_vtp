"""Continuous traversability supervision from simulator depth and semantics.

What this produces
------------------
For every frame, a dense target `y in [0, 1]` written as an 8-bit PNG. The value
is not a probability and not a class -- it is *expected cost-free progress* for a
stated reference platform:

    1.00  engineered hard flat surface
    0.85  natural but compacted, or hard with joints
    0.70  loose or vegetated but reliably drivable
    0.50  marginal: passable slowly or with risk
    0.30  strongly discouraged
    0.00  not traversable

The target is assembled as

    y = semantic_prior(class) x slope_factor x roughness_factor x height_gate

so semantics say what the surface *is* and geometry says what state it is *in*.
Neither alone suffices, which is the point: a concrete floor is 1.00 flat and
0.30 at 30 degrees, and grass and a tree trunk carry the same class label in
every forest environment in TartanGround.

Design decisions
----------------
1. The scale is absolute, not per-environment. A per-scene mapping makes part of
   the target unpredictable from the image -- the model would have to infer which
   environment it is in to know the offset -- and makes a deployment threshold
   chosen in one scene meaningless in another. See `taxonomy.py`.

2. Camera height is estimated once per trajectory, not per frame. It is a
   rigid-body constant, but a per-frame ground-pixel median swings by half a
   metre on vegetated terrain, which moves the height gate and intermittently
   blanks legitimate ground.

3. Slope is measured at terrain scale (an 11 px stencil, not 3). Depth resolves
   individual leaves; a narrow stencil measures the tilt of whichever leaf the
   pixel landed on rather than the grade of the ground.

4. Modulation is smoothed at sigma 12 and normalised by the traversable mask.
   The decoder predicts at stride 4, so finer detail is unrepresentable and only
   contributes an irreducible noise gradient. Normalising by the mask stops a
   wall -- legitimately near-vertical -- bleeding a dark halo onto the drivable
   ground beside it. Class boundaries stay perfectly sharp: only the modulation
   is smoothed, never the semantic prior.

   Together (2)-(4) cut frame-to-frame label delta by ~34% (up to 43% on the
   worst environments) while leaving mid-range pixel mass and mean label value
   essentially unchanged -- i.e. what was removed was noise, not signal.

5. Height and slope are gravity-aligned using per-frame pose. Treating the
   optical axis as horizontal makes the height error grow *linearly with range*
   -- about 0.87 m at 50 m for one degree of pitch -- so distant flat ground
   drifts upward until it crosses the height gate and is labelled an obstacle.
   Without a pose file the code falls back to the level-camera assumption and
   says so, rather than silently producing wrong geometry.

6. Vegetation is graded by height. One class covers grass, bushes, trunks and
   canopy in the forest environments (`instancedfoliageactor` alone is 56% of
   Gascola's pixels), so no single prior is right. Grading is only viable
   because of (5): a height threshold is meaningless if height drifts with range.

Expected input layout (TartanGround)
------------------------------------
    <root>/<Env>/seg_label_map.json
    <root>/<Env>/Data_diff/<Pxxxx>/depth_lcam_front/000000_lcam_front_depth.png
    <root>/<Env>/Data_diff/<Pxxxx>/seg_lcam_front/000000_lcam_front_seg.png
    <root>/<Env>/Data_diff/<Pxxxx>/pose_lcam_front.txt          (optional)
    <root>/<Env>/Data_diff/<Pxxxx>/<Pxxxx>_metadata.json        (optional)

Output: `<traj>/trav_lcam_front/{frame:06d}_lcam_front_trav.png`

    python -m sam_vtp_core.labels.generate --root /data/tartanground --workers 24
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import traceback
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from . import taxonomy as tax

# ---------------------------------------------------------------------------
# Camera. TartanGround lcam_front: 640x640 at 90 deg on both axes.
# ---------------------------------------------------------------------------
IMG_HW = 640
FX = FY = IMG_HW / 2.0
CX = CY = IMG_HW / 2.0

# ---------------------------------------------------------------------------
# Reference platform. Mid-size wheeled/tracked UGV: ~0.15 m wheel radius,
# ~0.20 m clearance, ~30 deg stable slope, ~0.15 m negotiable step.
# ---------------------------------------------------------------------------
SLOPE_FREE_DEG = 12.0       # no penalty below this
SLOPE_LIMIT_DEG = 35.0      # full penalty at or above this
ROUGH_FREE_M = 0.03         # local height std below which a surface is smooth
ROUGH_LIMIT_M = 0.20        # local height std at which the penalty saturates

# An obstacle is terrain standing above its *local* surroundings, which is
# invariant to overall grade; a flat-plane test would confuse a 20 deg uphill
# with a wall. The global cuts only reject clearly unreachable structure.
LOCAL_OBSTACLE_M = 0.55
LOCAL_TERRAIN_WINDOW = 65   # px
MAX_HEIGHT_M = 2.50         # overpasses, raised decking
MIN_HEIGHT_M = -1.00        # pits, water

ROUGHNESS_WINDOW = 9        # px
SLOPE_STENCIL = 11          # px half-width of the normal stencil
DEPTH_MEDIAN_K = 5          # px speckle rejection (cv2 float32 kernel limit)

# Floors on the geometric multipliers. Depth-derived slope and roughness are only
# meaningful where the sensor sees a coherent surface; on vegetation they report
# 20-50 deg on plainly drivable ground. The semantic class already carries most
# of the vegetation signal, so geometry refines rather than overrides it. Hard
# zeroing is reserved for the height constraint, which is a direct measurement.
SLOPE_FACTOR_FLOOR = 0.30
ROUGH_FACTOR_FLOOR = 0.40

MODULATION_SMOOTH_SIGMA = 12.0
CAMERA_HEIGHT_SAMPLES = 80  # frames sampled per trajectory to fix camera height

# Vegetation height bands, metres above ground, grounded in the same platform as
# the taxonomy: ground cover the wheels ride over, low bush that is passable but
# costly, and standing growth that is not passable at all.
VEG_GROUNDCOVER_M = 0.25
VEG_OBSTRUCTION_M = 0.60    # just above LOCAL_OBSTACLE_M, so the two agree

LEVEL_CAMERA_GRAVITY = np.array([0.0, 1.0, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def unproject(depth: np.ndarray):
    """Pinhole unprojection of planar z-depth into camera-frame XYZ (y down)."""
    h, w = depth.shape
    v, u = np.meshgrid(np.arange(h, dtype=np.float32),
                       np.arange(w, dtype=np.float32), indexing="ij")
    return (u - CX) * depth / FX, (v - CY) * depth / FY, depth


def gravity_in_optical(quat) -> np.ndarray:
    """Unit gravity direction in the camera's optical frame.

    `quat` is `[qx, qy, qz, qw]`, camera-to-world, in a NED world frame whose z
    axis points down. Optical axes are x right, y down, z forward, and the
    optical->NED body permutation is `ned = (opt_z, opt_x, opt_y)`.

    Only the third row of the rotation is needed; working the permutation
    through by hand gives `(R[2,1], R[2,2], R[2,0])`. For a level camera this
    returns `(0, 1, 0)` -- the optical y axis -- so the gravity-aligned pipeline
    reduces exactly to the level-camera one when the camera is level.
    """
    x, y, z, w = (float(q) for q in quat)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if not np.isfinite(n) or n < 1e-12:
        return LEVEL_CAMERA_GRAVITY.copy()
    x, y, z, w = x / n, y / n, z / n, w / n
    g = np.array([2.0 * (y * z + w * x),
                  1.0 - 2.0 * (x * x + y * y),
                  2.0 * (x * z - w * y)], dtype=np.float32)
    return g / max(1e-12, float(np.linalg.norm(g)))


def down_component(depth: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Distance below the camera along true gravity, for every pixel."""
    x, y, z = unproject(depth)
    return (g[0] * x + g[1] * y + g[2] * z).astype(np.float32)


def surface_slope_deg(depth: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Slope angle from depth-derived surface normals, measured against gravity.

    Depth is median-filtered to reject speckle, and the normal is differenced
    over a wide stencil so the measurement reflects the grade of the terrain
    rather than the tilt of individual foliage. (Gaussian pre-blur would invent
    phantom slopes across depth discontinuities, so median it is.)
    """
    d = np.ascontiguousarray(depth, dtype=np.float32)
    if DEPTH_MEDIAN_K > 1:
        d = cv2.medianBlur(d, DEPTH_MEDIAN_K)

    x, y, z = unproject(d)
    pts = np.stack([x, y, z], axis=-1)

    k = SLOPE_STENCIL
    gx = np.zeros_like(pts)
    gx[:, k:-k] = (pts[:, 2 * k:] - pts[:, : -2 * k]) / (2.0 * k)
    gy = np.zeros_like(pts)
    gy[k:-k, :] = (pts[2 * k:, :] - pts[: -2 * k, :]) / (2.0 * k)

    n = np.cross(gx, gy)
    norm = np.linalg.norm(n, axis=-1)
    valid = (norm > 1e-6) & (z > 0) & np.isfinite(z)

    cos_t = np.zeros_like(norm)
    dot = np.abs(n[..., 0] * g[0] + n[..., 1] * g[1] + n[..., 2] * g[2])
    np.divide(dot, norm, out=cos_t, where=valid)
    slope = np.degrees(np.arccos(np.clip(cos_t, 0.0, 1.0)))
    return np.where(valid, slope, np.nan).astype(np.float32)


def _valid_box_mean(field: np.ndarray, window: int):
    """Box mean ignoring NaNs. Returns (mean, valid_count)."""
    f = np.nan_to_num(field, nan=0.0).astype(np.float32)
    valid = np.isfinite(field).astype(np.float32)
    k = (window, window)
    s = cv2.boxFilter(f * valid, -1, k, normalize=False)
    n = cv2.boxFilter(valid, -1, k, normalize=False)
    return s / np.maximum(n, 1.0), n


def height_above_local_terrain(height: np.ndarray,
                               window: int = LOCAL_TERRAIN_WINDOW) -> np.ndarray:
    """How far each pixel stands above the local terrain trend, in metres.

    High-pass filtering the height field makes this invariant to a smooth grade:
    a uniform slope contributes ~0, while rocks, walls and vehicles standing
    proud of the surface produce a large positive response.
    """
    trend, n = _valid_box_mean(height, window)
    return np.where(n >= 16, height - trend, np.nan).astype(np.float32)


def local_roughness(height: np.ndarray, window: int = ROUGHNESS_WINDOW) -> np.ndarray:
    """Local standard deviation of height, in metres.

    Separates a smooth ramp (high slope, low roughness) from rubble (moderate
    slope, high roughness); the two are traversed very differently.
    """
    h = np.nan_to_num(height, nan=0.0).astype(np.float32)
    valid = np.isfinite(height).astype(np.float32)
    k = (window, window)
    s1 = cv2.boxFilter(h * valid, -1, k, normalize=False)
    s2 = cv2.boxFilter(h * h * valid, -1, k, normalize=False)
    n = cv2.boxFilter(valid, -1, k, normalize=False)
    n_safe = np.maximum(n, 1.0)
    mean = s1 / n_safe
    var = np.maximum(s2 / n_safe - mean * mean, 0.0)
    return np.where(n >= 4, np.sqrt(var), np.nan).astype(np.float32)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def _ramp(value, free, limit, floor=0.0):
    """1.0 below `free`, `floor` at or above `limit`, linear between.

    Unmeasurable pixels (NaN) fall back to 1.0, so a missing geometric cue never
    penalises the semantic prior.
    """
    out = 1.0 - (value - free) / max(1e-6, (limit - free))
    out = floor + (1.0 - floor) * np.clip(out, 0.0, 1.0)
    return np.where(np.isfinite(value), out, 1.0)


def _smooth_modulation(mod: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth the modulation, normalised by the traversable mask."""
    if sigma <= 0:
        return mod
    m = mod.astype(np.float32)
    w = weight.astype(np.float32)
    num = cv2.GaussianBlur(m * w, (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-6), m).astype(np.float32)


def vegetation_prior(height: np.ndarray) -> np.ndarray:
    """Traversability of ambiguous vegetation as a function of how tall it stands.

    Ramps rather than steps, so a bush does not change category over one pixel of
    depth noise. Pixels with no measurable height keep their original class prior
    (handled by the caller): an unknown height is not evidence of a tree.
    """
    t = np.clip((height - VEG_GROUNDCOVER_M) / (VEG_OBSTRUCTION_M - VEG_GROUNDCOVER_M), 0.0, 1.0)
    graded = tax.VEGETATED + (tax.DENSE_LOW_VEG - tax.VEGETATED) * t
    return np.where(height > VEG_OBSTRUCTION_M, 0.0, graded).astype(np.float32)


def build_veg_lut(name_map: dict) -> np.ndarray:
    """Which pixel values carry a height-ambiguous vegetation class.

    Defined as "resolves to DENSE_LOW_VEG" rather than as a hand-listed set, so
    a class added to the taxonomy at that tier is graded automatically and the
    two files cannot drift apart. That tier is exactly the ambiguous group:
    foliage, bush, shrub, hedge, plant, instancedfoliageactor.
    """
    out = np.zeros(256, dtype=bool)
    for class_name, pixel_value in name_map.items():
        if abs(tax.lookup(class_name, tax.BLOCKED) - tax.DENSE_LOW_VEG) < 1e-6:
            out[int(pixel_value) % 256] = True
    return out


def compute_traversability(semantic: np.ndarray, depth: np.ndarray, camera_height: float,
                           lut: np.ndarray, veg_lut: np.ndarray, g: np.ndarray) -> np.ndarray:
    """The continuous target for one frame.

    `camera_height` is the per-trajectory constant; passing it in rather than
    re-deriving it here is what removes frame-to-frame ground-plane jitter.
    """
    base = lut[semantic.astype(np.uint8)]

    down = down_component(depth, g)
    invalid = (depth <= 0) | (depth > 100) | ~np.isfinite(depth)
    down = np.where(invalid, np.nan, down)
    height = (camera_height - down).astype(np.float32)

    # Ambiguous vegetation is re-priced by height, but only where height is
    # known: a missing depth reading must not be read as evidence of a tree.
    gradeable = veg_lut[semantic.astype(np.uint8)] & np.isfinite(height)
    if gradeable.any():
        base = np.where(gradeable, vegetation_prior(height), base)

    slope = surface_slope_deg(depth, g)
    rough = local_roughness(height)

    modulation = (_ramp(slope, SLOPE_FREE_DEG, SLOPE_LIMIT_DEG, SLOPE_FACTOR_FLOOR)
                  * _ramp(rough, ROUGH_FREE_M, ROUGH_LIMIT_M, ROUGH_FACTOR_FLOOR))
    modulation = _smooth_modulation(modulation, base > 0.0, MODULATION_SMOOTH_SIGMA)
    trav = base * modulation

    valid_h = np.isfinite(height)
    local = height_above_local_terrain(height)
    trav = np.where(np.isfinite(local) & (local > LOCAL_OBSTACLE_M), 0.0, trav)
    trav = np.where(valid_h & (height > MAX_HEIGHT_M), 0.0, trav)
    trav = np.where(valid_h & (height < MIN_HEIGHT_M), 0.0, trav)

    # Pixels with no usable depth keep the semantic prior alone.
    return np.clip(np.nan_to_num(trav, nan=0.0), 0.0, 1.0).astype(np.float32)


def to_uint8(trav: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.clip(trav, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_name_map(root: Path, env: str) -> dict:
    for fn in ("seg_label_map.json", "seg_label.json"):
        p = root / env / fn
        if p.exists():
            try:
                return json.load(open(p)).get("name_map", {})
            except Exception:
                return {}
    return {}


def read_frame(depth_dir: Path, seg_dir: Path, i: int):
    """Read one (depth, semantic) pair. Depth is float32 packed into an RGBA PNG."""
    dp = depth_dir / f"{i:06d}_lcam_front_depth.png"
    sp = seg_dir / f"{i:06d}_lcam_front_seg.png"
    if not dp.exists() or not sp.exists():
        return None, None
    raw = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None, None
    depth = raw.view("<f4")
    depth = depth[:, :, 0] if depth.ndim == 3 else depth
    semantic = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE)
    if semantic is None:
        return None, None
    return depth.astype(np.float32), semantic


def load_poses(base_dir: Path) -> Optional[np.ndarray]:
    """Per-frame camera-to-world poses, `[x y z qx qy qz qw]` per row.

    Returns None when the file is absent, which makes the caller fall back to
    the level-camera assumption rather than silently producing wrong geometry.
    """
    p = base_dir / "pose_lcam_front.txt"
    if not p.exists():
        return None
    try:
        a = np.loadtxt(p, dtype=np.float64)
    except Exception:
        return None
    return a.reshape(1, -1) if a.ndim == 1 else a


def frame_camera_height(down: np.ndarray, depth: np.ndarray,
                        ground_mask: np.ndarray) -> float:
    """Single-frame estimate of camera height above the local ground plane.

    Anchored to the near field at the bottom-centre of the image. Using every
    drivable-looking pixel is not robust: elevated-but-drivable structure (a
    mill deck 2.2-2.6 m up) drags the plane upward until the actual ground reads
    negative and is discarded as a hole.

    Deliberately does not trust `robot_height` metadata, which overestimates the
    true camera-to-ground offset by ~1.2 m on some environments.
    """
    h, w = down.shape
    near = np.zeros((h, w), dtype=bool)
    near[int(0.55 * h):, int(0.20 * w):int(0.80 * w)] = True

    valid = np.isfinite(down) & (depth > 1.0) & (depth < 15.0) & near
    sel = valid & ground_mask
    if int(sel.sum()) < 1000:
        sel = valid                                     # any near-field surface
    if int(sel.sum()) < 500:
        sel = np.isfinite(down) & (depth > 0.5) & (depth < 40.0) & ground_mask
    if int(sel.sum()) < 500:
        return float("nan")

    est = float(np.median(down[sel]))
    if not np.isfinite(est) or est <= 0.05 or est > 5.0:
        return float("nan")
    return est


def estimate_trajectory_camera_height(depth_dir: Path, seg_dir: Path, poses, n_frames: int,
                                      lut: np.ndarray, fallback: float
                                      ) -> Tuple[float, int, float]:
    """One camera height for the whole trajectory: median of sampled frames.

    Returns `(height, n_used, mad)`. The median absolute deviation is reported so
    a trajectory whose geometry genuinely does not support a single plane is
    visible rather than silently averaged.
    """
    if n_frames <= 0:
        return float(fallback), 0, float("nan")
    idx = np.unique(np.linspace(0, n_frames - 1,
                                min(CAMERA_HEIGHT_SAMPLES, n_frames)).astype(int))
    ests = []
    for i in idx:
        depth, semantic = read_frame(depth_dir, seg_dir, int(i))
        if depth is None:
            continue
        g = (gravity_in_optical(poses[i][3:]) if poses is not None and i < len(poses)
             else LEVEL_CAMERA_GRAVITY)
        base = lut[semantic.astype(np.uint8)]
        down = down_component(depth, g)
        invalid = (depth <= 0) | (depth > 100) | ~np.isfinite(depth)
        down = np.where(invalid, np.nan, down)
        e = frame_camera_height(down, depth, base >= tax.FIRM_NATURAL)
        if np.isfinite(e):
            ests.append(e)
    if not ests:
        return float(fallback), 0, float("nan")
    a = np.asarray(ests, dtype=np.float64)
    med = float(np.median(a))
    return med, len(ests), float(np.median(np.abs(a - med)))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def process_sequence(root: Path, env: str, seq: str, out_name: str = "trav_lcam_front",
                     overwrite: bool = False, max_frames: Optional[int] = None) -> Tuple[int, list]:
    """Generate labels for one trajectory. Returns (frames written, notes)."""
    notes = []
    base_dir = root / env / "Data_diff" / seq
    depth_dir, seg_dir = base_dir / "depth_lcam_front", base_dir / "seg_lcam_front"
    out_dir = base_dir / out_name
    if not depth_dir.exists() or not seg_dir.exists():
        return 0, [f"{env}/{seq}: no depth or seg directory"]

    name_map = load_name_map(root, env)
    if not name_map:
        return 0, [f"{env}/{seq}: no semantic label map"]
    lut = tax.build_lut(name_map)
    veg_lut = build_veg_lut(name_map)

    # Only a fallback, for a trajectory with too few ground pixels to self-calibrate.
    robot_height = 0.5
    meta_path = base_dir / f"{seq}_metadata.json"
    if meta_path.exists():
        try:
            robot_height = float(json.load(open(meta_path))["robot_height"])
        except Exception:
            pass

    depth_files = sorted(depth_dir.glob("*_lcam_front_depth.png"))
    if max_frames is not None:
        depth_files = depth_files[:max_frames]

    poses = load_poses(base_dir)
    if poses is None:
        notes.append(f"{env}/{seq}: NO POSES -- falling back to the level-camera "
                     f"assumption for this trajectory")
    elif len(poses) < len(depth_files):
        notes.append(f"{env}/{seq}: {len(poses)} poses for {len(depth_files)} frames; "
                     f"frames beyond the pose file use the level-camera assumption")

    cam_h, n_used, mad = estimate_trajectory_camera_height(
        depth_dir, seg_dir, poses, len(depth_files), lut, robot_height)
    if n_used == 0:
        notes.append(f"{env}/{seq}: no usable ground pixels, falling back to "
                     f"metadata {robot_height:.3f} m")
    elif np.isfinite(mad) and mad > 0.15:
        notes.append(f"{env}/{seq}: camera height {cam_h:.3f} m but MAD {mad:.3f} m "
                     f"over {n_used} frames")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for i in range(len(depth_files)):
        out_path = out_dir / f"{i:06d}_lcam_front_trav.png"
        if out_path.exists() and not overwrite:
            continue
        depth, semantic = read_frame(depth_dir, seg_dir, i)
        if depth is None:
            continue
        g = (gravity_in_optical(poses[i][3:]) if poses is not None and i < len(poses)
             else LEVEL_CAMERA_GRAVITY)
        cv2.imwrite(str(out_path),
                    to_uint8(compute_traversability(semantic, depth, cam_h, lut, veg_lut, g)))
        written += 1
    return written, notes


def _job(args):
    root, env, seq, out_name, overwrite, max_frames = args
    try:
        n, notes = process_sequence(root, env, seq, out_name, overwrite, max_frames)
        return env, seq, n, notes, None
    except Exception:                        # one bad sequence must not abandon
        return env, seq, 0, [], traceback.format_exc(limit=2).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="TartanGround dataset root")
    ap.add_argument("--envs", nargs="*", default=None, help="default: every environment found")
    ap.add_argument("--out-name", default="trav_lcam_front",
                    help="subdirectory written inside each trajectory")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                    help="parallel trajectories; each is independent, having its own camera height")
    a = ap.parse_args()

    root: Path = a.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"dataset root not found: {root}")

    envs = a.envs or sorted(p.name for p in root.iterdir()
                            if p.is_dir() and (p / "Data_diff").is_dir())
    jobs = []
    for env in envs:
        d = root / env / "Data_diff"
        if not d.is_dir():
            continue
        for seq in sorted(p.name for p in d.iterdir() if p.is_dir()):
            if (d / seq / "depth_lcam_front").is_dir():
                jobs.append((root, env, seq, a.out_name, a.overwrite, a.max_frames))
    if not jobs:
        raise SystemExit(f"no trajectories found under {root}")

    # Each worker is numpy/OpenCV heavy; letting both libraries also thread
    # internally oversubscribes the machine and runs slower than one thread each.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    cv2.setNumThreads(1)

    print(f"{len(jobs)} trajectories over {len(envs)} environments, {a.workers} workers")
    total, failed = 0, []
    with mp.Pool(a.workers) as pool:
        for i, (env, seq, n, notes, err) in enumerate(pool.imap_unordered(_job, jobs), 1):
            total += n
            for note in notes:
                print(f"  {note}")
            if err:
                failed.append((env, seq, err))
                print(f"  FAILED {env}/{seq}: {err.splitlines()[-1]}")
            if i % 10 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] trajectories, {total:,} frames")

    print(f"\nwrote {total:,} traversability maps; {len(failed)} trajectories failed")
    for e, s, err in failed:
        print(f"  {e}/{s}: {err.splitlines()[-1]}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
