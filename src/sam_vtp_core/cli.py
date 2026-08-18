"""`sam-vtp-infer`: predict traversability for one image sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .factory import build_model, describe
from .inference import infer_sequence, list_depths, list_images


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run SAM-VTP over a temporally ordered folder of frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Frames are consumed in sorted filename order and memory is carried "
            "across them, so the folder must be one sequence in capture order.\n"
            "Released checkpoints use depth; pass --depth-dir."
        ))
    p.add_argument("--images-dir", required=True, help="folder of RGB frames, one sequence")
    p.add_argument("--output-dir", required=True, help="where to write probability/ and binary/")
    p.add_argument("--checkpoint", "--sam-vtp-checkpoint", dest="checkpoint", required=True)

    p.add_argument("--depth-dir", default=None,
                   help="folder of depth maps aligned with --images-dir (.npy or .png)")
    p.add_argument("--depth-scale", type=float, default=1.0,
                   help="multiplier converting stored depth to METRES (0.001 for millimetres)")
    p.add_argument("--height", type=int, default=0, help="resize height; 0 keeps native size")
    p.add_argument("--width", type=int, default=0, help="resize width; 0 keeps native size")
    p.add_argument("--threshold", type=float, default=0.5, help="binarisation threshold")
    p.add_argument("--normalization", default="0_255", choices=["0_255", "0_1", "-1_1", "auto"])
    p.add_argument("--memory-write-every", type=int, default=1,
                   help="write memory every k-th frame; raises the bank's time span on fast cameras")
    p.add_argument("--amp", choices=["fp16", "fp32"], default="fp16")

    # Component switches. These select which of the checkpoint's weights are
    # used; they cannot add a component it was not trained with. Any deviation
    # from the as-trained configuration is printed by describe() below, so an
    # ablation run is never mistaken for a full-model run in a log.
    p.add_argument("--no-memory", dest="memory", action="store_false", default=None,
                   help="frame-wise: predict each frame independently, no temporal memory")
    p.add_argument("--no-geometry", dest="geometry", action="store_false", default=None,
                   help="RGB-only: drop the depth-derived feature")
    p.add_argument("--no-refine", dest="refine", action="store_false", default=None,
                   help="skip the residual refinement head")

    p.add_argument("--use-fullgraph", dest="use_fullgraph", action="store_true", default=None,
                   help="CUDA graph replay of the full forward pass; default on when CUDA is available")
    p.add_argument("--no-fullgraph", dest="use_fullgraph", action="store_false")
    p.add_argument("--capture-warmup", type=int, default=0)
    p.add_argument("--fullgraph-min-history", type=int, default=16)

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cuda", "cpu"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device)

    switches = {k: getattr(args, k) for k in ("memory", "geometry", "refine")
                if getattr(args, k) is not None}

    model = build_model(
        args.checkpoint, device=device,
        normalization_mode=args.normalization,
        memory_write_every=int(args.memory_write_every),
        components=switches or None,
    )
    print(describe(model))

    image_paths = list_images(args.images_dir)
    depth_paths = list_depths(args.depth_dir) if args.depth_dir else None

    resize_hw = None
    if args.height > 0 and args.width > 0:
        resize_hw = (int(args.height), int(args.width))

    use_fullgraph = (args.use_fullgraph if args.use_fullgraph is not None
                     else device.type == "cuda")

    stats = infer_sequence(
        model=model, image_paths=image_paths, output_dir=args.output_dir,
        resize_hw=resize_hw, threshold=float(args.threshold),
        depth_paths=depth_paths, depth_scale=float(args.depth_scale),
        use_fullgraph=bool(use_fullgraph),
        fullgraph_min_history=int(args.fullgraph_min_history),
        capture_warmup=int(args.capture_warmup), amp_dtype=args.amp)

    print(f"frames       : {stats['num_frames']}")
    print(f"total seconds: {stats['total_seconds']:.3f}")
    print(f"average FPS  : {stats['fps']:.2f}")
    print(f"graph status : {stats['graph_status']}")
    print(f"outputs      : {Path(args.output_dir).expanduser().resolve()}")
    return 0
