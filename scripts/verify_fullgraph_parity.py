#!/usr/bin/env python3
"""Check that CUDA-graph replay agrees with the eager forward pass.

`fullgraph.py` is a second implementation of the forward pass, written over
tensorised state so it can be captured. Two implementations drift. This script
runs the same sequence through both and reports the largest disagreement, so a
change to `architecture.py` that the graph path did not follow shows up as a
number rather than as quietly different published latency-mode results.

Graph replay only covers RGB-only models, so the geometry path is switched off
here regardless of what the checkpoint carries -- this validates the graph
implementation, not the released configuration.

    python scripts/verify_fullgraph_parity.py --checkpoint sam_vtp.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam_vtp_core import SAMVTP, SAMVTPConfig, build_model  # noqa: E402
from sam_vtp_core.fullgraph import SAMVTPFullGraphReplayRunner  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="", help="omit to use random weights")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--tolerance", type=float, default=1e-3)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; graph replay cannot be tested here.")
        return 77
    device = torch.device("cuda")

    def make():
        if a.checkpoint:
            m = build_model(a.checkpoint, device=device)
        else:
            m = SAMVTP(SAMVTPConfig(warp_memory=False, use_geometry=False)).to(device).eval()
        # Graph replay is RGB-only; disabling here is what makes the comparison
        # meaningful rather than a comparison of two eager runs.
        m.warp_memory = False
        m.use_geometry = False
        return m

    torch.manual_seed(0)
    frames = [torch.randint(0, 256, (1, 3, a.size, a.size), device=device).float()
              for _ in range(a.frames)]

    eager_model = make()
    with torch.no_grad():
        eager = [eager_model.forward_frame(f, is_init_frame=(i == 0))["logits"].clone()
                 for i, f in enumerate(frames)]

    graph_model = make()
    runner = SAMVTPFullGraphReplayRunner(graph_model, device, enabled=True, capture_warmup=1)
    with torch.no_grad():
        graphed = [runner.run(f, is_init_frame=(i == 0))["logits"].clone()
                   for i, f in enumerate(frames)]

    print(runner.status_note())
    worst, worst_t = 0.0, -1
    for t, (e, g) in enumerate(zip(eager, graphed)):
        d = (e - g).abs().max().item()
        if d > worst:
            worst, worst_t = d, t
    print(f"frames compared : {len(eager)}")
    print(f"max |dlogit|    : {worst:.3e} (frame {worst_t})")
    ok = worst <= a.tolerance
    print("PARITY OK" if ok else f"PARITY FAILED (tolerance {a.tolerance:.1e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
