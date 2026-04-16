from __future__ import annotations

from contextlib import nullcontext
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .fullgraph import SAMVTPFullGraphReplayRunner
from .model import SAMVTP


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _conv_out_size(size: int, kernel: int, stride: int, padding: int) -> int:
    return ((int(size) + 2 * int(padding) - int(kernel)) // int(stride)) + 1


def _min_input_for_conv_out(out_size: int, kernel: int, stride: int, padding: int) -> int:
    # Smallest integer input size that produces at least `out_size` after the conv.
    return ((int(out_size) - 1) * int(stride)) - 2 * int(padding) + int(kernel)


def _sam2_compatible_input_hw(model: SAMVTP, in_h: int, in_w: int) -> Tuple[int, int]:
    """Return nearest-up input size compatible with SAM2 Hiera window positional tiling."""
    trunk = model.sam2_model.image_encoder.trunk
    patch_proj = trunk.patch_embed.proj

    s_h, s_w = int(patch_proj.stride[0]), int(patch_proj.stride[1])
    p_h, p_w = int(patch_proj.padding[0]), int(patch_proj.padding[1])
    k_h, k_w = int(patch_proj.kernel_size[0]), int(patch_proj.kernel_size[1])

    window = int(trunk.window_spec[0]) if hasattr(trunk, "window_spec") else 1
    window = max(1, window)

    tok_h = _conv_out_size(in_h, kernel=k_h, stride=s_h, padding=p_h)
    tok_w = _conv_out_size(in_w, kernel=k_w, stride=s_w, padding=p_w)

    tok_h_compat = ((tok_h + window - 1) // window) * window
    tok_w_compat = ((tok_w + window - 1) // window) * window

    req_h = max(int(in_h), _min_input_for_conv_out(tok_h_compat, kernel=k_h, stride=s_h, padding=p_h))
    req_w = max(int(in_w), _min_input_for_conv_out(tok_w_compat, kernel=k_w, stride=s_w, padding=p_w))
    return req_h, req_w


def list_images(image_dir: str) -> List[Path]:
    root = Path(image_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError("Image directory not found: {}".format(root))
    if not root.is_dir():
        raise NotADirectoryError("Expected a directory, got: {}".format(root))

    paths = [p for p in sorted(root.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not paths:
        raise RuntimeError("No images found in {}".format(root))
    return paths


def load_frame_tensor(image_path: Path, resize_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    if resize_hw is not None:
        out_h, out_w = int(resize_hw[0]), int(resize_hw[1])
        img = img.resize((out_w, out_h), Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32)
    chw = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return chw.unsqueeze(0)


def save_probability_png(probability_map: np.ndarray, output_path: Path) -> None:
    clipped = np.clip(probability_map, 0.0, 1.0)
    arr_u8 = (clipped * 255.0).astype(np.uint8)
    Image.fromarray(arr_u8, mode="L").save(output_path)


def save_binary_png(probability_map: np.ndarray, threshold: float, output_path: Path) -> None:
    binary = (probability_map >= float(threshold)).astype(np.uint8) * 255
    Image.fromarray(binary, mode="L").save(output_path)


def infer_sequence(
    model: SAMVTP,
    image_paths: Sequence[Path],
    output_dir: Optional[str] = None,
    resize_hw: Optional[Tuple[int, int]] = None,
    threshold: float = 0.5,
    use_rolling_memory: bool = True,
    use_fullgraph: bool = False,
    fullgraph_min_history: int = 16,
    capture_warmup: int = 0,
    amp_dtype: str = "fp16",
) -> dict:
    if not image_paths:
        raise ValueError("image_paths must not be empty")

    device = next(model.parameters()).device
    runner = SAMVTPFullGraphReplayRunner(
        model=model,
        device=device,
        enabled=use_fullgraph,
        capture_warmup=capture_warmup,
        min_history_frames=fullgraph_min_history,
    )

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
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            out_h, out_w = int(frame.shape[-2]), int(frame.shape[-1])
            proc_h, proc_w = _sam2_compatible_input_hw(model, out_h, out_w)
            if (proc_h != out_h) or (proc_w != out_w):
                # Replicate-border padding preserves edge content and keeps prompts meaningful.
                frame = F.pad(frame, (0, proc_w - out_w, 0, proc_h - out_h), mode="replicate")

            is_init = idx == 0
            t0 = time.perf_counter()
            amp_context = (
                torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                if use_cuda_amp
                else nullcontext()
            )
            with amp_context:
                if use_fullgraph:
                    out = runner.run(frame, is_init_frame=is_init, use_rolling_memory=use_rolling_memory)
                else:
                    out = model.forward_frame(
                        frame,
                        is_init_frame=is_init,
                        use_rolling_memory=use_rolling_memory,
                        update_memory=True,
                    )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - t0)

            logits = out["logits"]
            if logits.shape[-2:] != (out_h, out_w):
                logits = logits[..., :out_h, :out_w]

            probs = torch.sigmoid(logits)[0, 0].detach().float().cpu().numpy()

            if output_root is not None:
                stem = path.stem
                save_probability_png(probs, output_root / "probability" / (stem + "_prob.png"))
                save_binary_png(probs, threshold=threshold, output_path=output_root / "binary" / (stem + "_bin.png"))

    total = float(np.sum(timings))
    fps = float(len(timings) / total) if total > 0.0 else 0.0
    return {
        "num_frames": len(image_paths),
        "total_seconds": total,
        "fps": fps,
        "graph_status": runner.status_note(),
    }
