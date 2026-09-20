"""`sam-vtp-smoke`: check that a checkpoint builds, loads and runs.

Runs a short synthetic sequence rather than a single frame, because the failures
worth catching -- a mismatched memory width, a wrongly inferred architecture,
missing depth -- only appear once memory is read back.
"""

from __future__ import annotations

import argparse

import torch

from .build import build_model, describe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM-VTP smoke test")
    p.add_argument("--checkpoint", "--sam-vtp-checkpoint", dest="checkpoint", required=True)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--device", choices=["cuda", "cpu"],
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    model = build_model(args.checkpoint, device=device)
    print(describe(model))

    T, H, W = int(args.frames), int(args.height), int(args.width)
    # Distinct frames, not one frame repeated: with identical inputs a broken
    # temporal path still produces a plausible sequence.
    g = torch.Generator().manual_seed(0)
    images = (torch.rand(1, T, 3, H, W, generator=g) * 255.0).to(device)

    depth = None
    if model.needs_depth:
        depth = (torch.rand(1, T, 1, H, W, generator=g) * 20.0 + 1.0).to(device)

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


if __name__ == "__main__":
    raise SystemExit(main())
