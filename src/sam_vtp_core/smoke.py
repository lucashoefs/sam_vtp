"""`sam-vtp-smoke`: check that a checkpoint builds, loads and runs.

Runs a short synthetic sequence rather than a single frame, because the failures
worth catching -- a mismatched memory width, a wrongly inferred architecture,
missing depth -- only appear once memory is read back.
"""

from __future__ import annotations

import argparse

import torch

from .architecture import SAMVTP, SAMVTPConfig
from .factory import build_model, describe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM-VTP smoke test")
    p.add_argument("--checkpoint", "--sam-vtp-checkpoint", dest="checkpoint", default="",
                   help="checkpoint to load; omit to test a randomly initialised model")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--device", choices=["cuda", "cpu"],
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        model = build_model(args.checkpoint, device=device)
    else:
        model = SAMVTP(SAMVTPConfig()).to(device).eval()
    print(describe(model))

    T, H, W = int(args.frames), int(args.height), int(args.width)
    images = torch.randint(0, 256, (1, T, 3, H, W), device=device).float()

    depth = None
    if model.needs_depth:
        depth = torch.rand(1, T, 1, H, W, device=device) * 20.0 + 1.0

    with torch.no_grad():
        out = model(images, depth_sequence=depth)

    logits = out["logits"]
    probs = torch.sigmoid(logits)
    print(f"logits {tuple(logits.shape)}  range [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"mean traversability per frame: "
          f"{[round(float(v), 4) for v in probs.mean(dim=(0, 2, 3, 4))]}")
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite logits")
    return 0
