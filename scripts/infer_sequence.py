#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sam_vtp_core import build_model, infer_sequence, list_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM-VTP inference on an image sequence folder.")
    parser.add_argument("--images-dir", type=str, required=True, help="Folder containing RGB frames.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output folder for masks.")

    parser.add_argument(
        "--sam2-cfg",
        type=str,
        default="configs/sam2.1_hiera_t.yaml",
        help="Path to SAM2 hydra model config.",
    )
    parser.add_argument(
        "--sam-vtp-checkpoint",
        type=str,
        required=True,
        help="Path to SAM-VTP checkpoint.",
    )

    parser.add_argument("--num-maskmem", type=int, default=7)
    parser.add_argument("--height", type=int, default=0, help="Optional resize height. Keep 0 to disable resize.")
    parser.add_argument("--width", type=int, default=0, help="Optional resize width. Keep 0 to disable resize.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp", choices=["fp16", "fp32"], default="fp16")

    parser.add_argument(
        "--use-fullgraph",
        dest="use_fullgraph",
        action="store_true",
        default=None,
        help="Enable CUDA graph full-forward replay. Default: enabled on CUDA.",
    )
    parser.add_argument(
        "--no-fullgraph",
        dest="use_fullgraph",
        action="store_false",
        help="Disable CUDA graph replay even when CUDA is available.",
    )
    parser.add_argument("--capture-warmup", type=int, default=0)
    parser.add_argument("--fullgraph-min-history", type=int, default=16)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested but not available.")

    model = build_model(
        sam2_cfg_path=args.sam2_cfg,
        sam_vtp_checkpoint_path=args.sam_vtp_checkpoint,
        device=device,
        num_maskmem=int(args.num_maskmem),
        freeze_encoder=True,
        freeze_decoder=False,
        multimask=True,
        normalization_mode="0_255",
    )

    image_paths = list_images(args.images_dir)

    resize_hw = None
    if args.height > 0 and args.width > 0:
        resize_hw = (int(args.height), int(args.width))

    use_fullgraph = args.use_fullgraph if args.use_fullgraph is not None else (device.type == "cuda")

    stats = infer_sequence(
        model=model,
        image_paths=image_paths,
        output_dir=args.output_dir,
        resize_hw=resize_hw,
        threshold=float(args.threshold),
        use_rolling_memory=True,
        use_fullgraph=bool(use_fullgraph),
        fullgraph_min_history=int(args.fullgraph_min_history),
        capture_warmup=int(args.capture_warmup),
        amp_dtype=args.amp,
    )

    print("=" * 80)
    print("SAM-VTP INFERENCE COMPLETE")
    print("=" * 80)
    print("Frames       : {}".format(stats["num_frames"]))
    print("Total sec    : {:.3f}".format(stats["total_seconds"]))
    print("Avg FPS      : {:.2f}".format(stats["fps"]))
    print("Graph status : {}".format(stats["graph_status"]))
    print("Outputs      : {}".format(Path(args.output_dir).expanduser().resolve()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
