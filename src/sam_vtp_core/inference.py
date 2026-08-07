"""Running SAM-VTP over a folder of frames."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .architecture import SAMVTP
from .fullgraph import SAMVTPFullGraphReplayRunner

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEPTH_EXTENSIONS = {".npy", ".png", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Input geometry
# ---------------------------------------------------------------------------
def _conv_out_size(size: int, kernel: int, stride: int, padding: int) -> int:
    return ((int(size) + 2 * int(padding) - int(kernel)) // int(stride)) + 1


def _min_input_for_conv_out(out_size: int, kernel: int, stride: int, padding: int) -> int:
    return ((int(out_size) - 1) * int(stride)) - 2 * int(padding) + int(kernel)


def compatible_input_hw(model: SAMVTP, in_h: int, in_w: int) -> Tuple[int, int]:
    """Smallest size >= (in_h, in_w) whose token grid tiles the Hiera window.

    The windowed attention partitions the token grid into fixed squares; a grid
    that is not a multiple of the window is padded internally in a way that
    shifts the trailing tokens. Padding the *image* instead keeps every token
    where the network expects it, and the logits are cropped back afterwards.
    """
    trunk = model.sam2_model.image_encoder.trunk
    proj = trunk.patch_embed.proj
    s_h, s_w = int(proj.stride[0]), int(proj.stride[1])
    p_h, p_w = int(proj.padding[0]), int(proj.padding[1])
    k_h, k_w = int(proj.kernel_size[0]), int(proj.kernel_size[1])
    window = max(1, int(trunk.window_spec[0]))

    tok_h = _conv_out_size(in_h, k_h, s_h, p_h)
    tok_w = _conv_out_size(in_w, k_w, s_w, p_w)
    tok_h = ((tok_h + window - 1) // window) * window
    tok_w = ((tok_w + window - 1) // window) * window

    return (max(int(in_h), _min_input_for_conv_out(tok_h, k_h, s_h, p_h)),
            max(int(in_w), _min_input_for_conv_out(tok_w, k_w, s_w, p_w)))


_sam2_compatible_input_hw = compatible_input_hw  # backwards-compatible alias


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def list_images(image_dir: str) -> List[Path]:
    root = Path(image_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    paths = [p for p in sorted(root.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not paths:
        raise RuntimeError(f"no images found in {root}")
    return paths


def list_depths(depth_dir: str) -> List[Path]:
    root = Path(depth_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    paths = [p for p in sorted(root.iterdir()) if p.suffix.lower() in DEPTH_EXTENSIONS]
    if not paths:
        raise RuntimeError(f"no depth maps found in {root}")
    return paths


def load_frame_tensor(image_path: Path, resize_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    if resize_hw is not None:
        img = img.resize((int(resize_hw[1]), int(resize_hw[0])), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0)


def read_depth_metres(path: Path, scale: float = 1.0, packed_bgra: bool = True) -> np.ndarray:
    """Read one depth map and return float32 METRES.

    Handles three encodings:
      * `.npy` -- float array, already metres (times `scale`). Prefer this: it
        is the only one of the three with no ambiguity.
      * 4-channel 8-bit PNG -- float32 packed into the colour channels, the
        TartanAir/TartanGround convention.
      * single-channel 16-bit PNG/TIFF -- integer units; pass `scale=0.001` for
        the millimetre convention most RGB-D cameras use.

    Two traps, both of which produce a plausible-looking wrong answer rather
    than an error:

    `scale`. The geometry channels and the ego-motion warp are metric, so depth
    in millimetres yields slopes, heights and correspondences wrong by a
    thousandfold while the network still emits a believable mask.

    `packed_bgra`. TartanAir's depth PNGs are written and read with OpenCV,
    which orders channels BGRA, whereas PIL returns the file's RGBA. Viewing
    those same bytes as float32 in the two orders gives *different depths, both
    in a believable range* -- on a real frame, 0.12-65504 m against
    0.03-32992 m. The default reproduces OpenCV's reading, which is what the
    labels and the trained model assume. Set it False only for float32 you
    packed yourself in RGBA order.
    """
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.float32) * float(scale)

    arr = np.asarray(Image.open(path))
    if arr.ndim == 3 and arr.shape[2] == 4 and arr.dtype == np.uint8:
        if packed_bgra:
            arr = arr[..., [2, 1, 0, 3]]
        return np.ascontiguousarray(arr).view("<f4")[:, :, 0].astype(np.float32) * float(scale)
    if arr.ndim != 2:
        raise ValueError(f"{path.name}: expected a single-channel depth map, got shape {arr.shape}")
    return arr.astype(np.float32) * float(scale)


def depth_from_uint16_mm(arr, scale: float = 0.001):
    """RealSense-style `16UC1` millimetres to float32 metres, plus a validity mask.

    Zero means "no return", not "zero distance", so it is reported as invalid
    rather than as a surface pressed against the lens. Stereo holes are routine
    on specular floors, low-texture surfaces and past the camera's usable range.

    Returns:
        (depth_metres, valid) as float32 and bool arrays of the input shape,
        ready for `forward_frame(depth=..., depth_valid=...)`.
    """
    a = np.asarray(arr)
    valid = a > 0
    return a.astype(np.float32) * float(scale), valid


def load_depth_tensor(path: Path, scale: float = 1.0,
                      resize_hw: Optional[Tuple[int, int]] = None,
                      packed_bgra: bool = True) -> torch.Tensor:
    d = torch.from_numpy(read_depth_metres(path, scale, packed_bgra))[None, None]
    if resize_hw is not None and d.shape[-2:] != tuple(resize_hw):
        # Nearest, deliberately: averaging across a depth discontinuity returns a
        # distance at which no surface exists.
        d = F.interpolate(d, size=(int(resize_hw[0]), int(resize_hw[1])), mode="nearest-exact")
    return d


def load_poses(path: str) -> torch.Tensor:
    """Load camera-to-world poses, one row of `[x y z qx qy qz qw]` per frame.

    The quaternion is scalar-LAST and the world frame is NED, matching
    TartanGround's `pose_lcam_front.txt`. A scalar-first file loads without
    complaint and rotates everything wrongly, so check your source.
    """
    arr = np.loadtxt(Path(path).expanduser(), dtype=np.float32)
    arr = arr.reshape(1, -1) if arr.ndim == 1 else arr
    if arr.shape[1] != 7:
        raise ValueError(f"expected 7 columns [x y z qx qy qz qw], got {arr.shape[1]}")
    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_probability_png(probability_map: np.ndarray, output_path: Path) -> None:
    arr = (np.clip(probability_map, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(output_path)


def save_binary_png(probability_map: np.ndarray, threshold: float, output_path: Path) -> None:
    arr = (probability_map >= float(threshold)).astype(np.uint8) * 255
    Image.fromarray(arr, mode="L").save(output_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def infer_sequence(
    model: SAMVTP,
    image_paths: Sequence[Path],
    output_dir: Optional[str] = None,
    resize_hw: Optional[Tuple[int, int]] = None,
    threshold: float = 0.5,
    depth_paths: Optional[Sequence[Path]] = None,
    poses: Optional[torch.Tensor] = None,
    depth_scale: float = 1.0,
    use_fullgraph: bool = False,
    fullgraph_min_history: int = 16,
    capture_warmup: int = 0,
    amp_dtype: str = "fp16",
) -> dict:
    """Predict traversability for one temporally ordered sequence.

    `image_paths` must be in capture order: the model carries memory across
    frames, so a shuffled folder produces a temporally incoherent result rather
    than an error.

    `depth_paths` and `poses` are required for checkpoints trained with the
    geometry path (all released ones). They must be aligned index-for-index
    with `image_paths`.
    """
    if not image_paths:
        raise ValueError("image_paths must not be empty")

    needs_geom = model.needs_geometry
    if needs_geom and (depth_paths is None or poses is None):
        raise ValueError(
            "this checkpoint uses depth and pose (warp_memory/use_geometry). Pass "
            "depth_paths and poses, or disable both on the model to run the "
            "RGB-only ablation deliberately.")
    if depth_paths is not None and len(depth_paths) != len(image_paths):
        raise ValueError(f"{len(depth_paths)} depth maps for {len(image_paths)} images")
    if poses is not None and len(poses) < len(image_paths):
        raise ValueError(f"{len(poses)} poses for {len(image_paths)} images")

    device = next(model.parameters()).device
    runner = SAMVTPFullGraphReplayRunner(
        model=model, device=device, enabled=use_fullgraph,
        capture_warmup=capture_warmup, min_history_frames=fullgraph_min_history)

    output_root = None
    if output_dir:
        output_root = Path(output_dir).expanduser().resolve()
        (output_root / "probability").mkdir(parents=True, exist_ok=True)
        (output_root / "binary").mkdir(parents=True, exist_ok=True)

    model.reset_memory()
    use_cuda_amp = (device.type == "cuda") and (amp_dtype == "fp16")

    timings = []
    with torch.no_grad():
        for idx, path in enumerate(image_paths):
            frame = load_frame_tensor(path, resize_hw=resize_hw).to(
                device=device, dtype=torch.float32, non_blocking=True)
            out_h, out_w = int(frame.shape[-2]), int(frame.shape[-1])

            depth = pose = None
            if depth_paths is not None:
                depth = load_depth_tensor(depth_paths[idx], scale=depth_scale,
                                          resize_hw=(out_h, out_w)).to(device, torch.float32)
            if poses is not None:
                pose = poses[idx : idx + 1].to(device, torch.float32)

            proc_h, proc_w = compatible_input_hw(model, out_h, out_w)
            if (proc_h, proc_w) != (out_h, out_w):
                # Replicate-border padding keeps edge content intact.
                frame = F.pad(frame, (0, proc_w - out_w, 0, proc_h - out_h), mode="replicate")
                if depth is not None:
                    depth = F.pad(depth, (0, proc_w - out_w, 0, proc_h - out_h), mode="replicate")

            t0 = time.perf_counter()
            amp_context = (torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                           if use_cuda_amp else nullcontext())
            with amp_context:
                if use_fullgraph:
                    out = runner.run(frame, is_init_frame=(idx == 0), depth=depth, pose=pose)
                else:
                    out = model.forward_frame(frame, is_init_frame=(idx == 0),
                                              depth=depth, pose=pose)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - t0)

            logits = out["logits"]
            if logits.shape[-2:] != (out_h, out_w):
                logits = logits[..., :out_h, :out_w]
            probs = torch.sigmoid(logits)[0, 0].detach().float().cpu().numpy()

            if output_root is not None:
                save_probability_png(probs, output_root / "probability" / f"{path.stem}_prob.png")
                save_binary_png(probs, threshold, output_root / "binary" / f"{path.stem}_bin.png")

    total = float(np.sum(timings))
    return {
        "num_frames": len(image_paths),
        "total_seconds": total,
        "fps": float(len(timings) / total) if total > 0 else 0.0,
        "graph_status": runner.status_note(),
    }
