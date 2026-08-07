#!/usr/bin/env python3
"""Check that CUDA-graph replay agrees with the eager forward pass.

`fullgraph.py` is a second implementation of the forward pass, written over
tensorised state so it can be captured. Two implementations drift. This runs the
same sequence through both and reports the largest disagreement, so a change to
`architecture.py` that the graph path did not follow shows up as a number rather
than as quietly different results whenever the speed flag is on.

Both are run in the model's own configuration, so a geometry checkpoint tests
the warp and geometry encoder inside the graph. `--rgb-only` forces the
ablation configuration instead.

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

from sam_vtp_core import SAMVTP, SAMVTPConfig, build_model, describe  # noqa: E402
from sam_vtp_core.fullgraph import SAMVTPFullGraphReplayRunner  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="", help="omit to use random weights")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--tolerance", type=float, default=1e-3)
    ap.add_argument("--rgb-only", action="store_true",
                    help="force warp_memory and use_geometry off")
    ap.add_argument("--memory-fraction", type=float, default=0.0,
                    help="cap this process's GPU memory, to share a busy device")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; graph replay cannot be tested here.")
        return 77
    device = torch.device("cuda")
    if a.memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction, 0)

    def make():
        if a.checkpoint:
            m = build_model(a.checkpoint, device=device)
        else:
            # Same seed for both copies, or this compares two different random
            # models and reports the difference as a capture failure.
            torch.manual_seed(1234)
            m = SAMVTP(SAMVTPConfig()).to(device).eval()
        if a.rgb_only:
            m.warp_memory = m.use_geometry = False
        return m

    torch.manual_seed(0)
    n = a.frames
    frames = [torch.randint(0, 256, (1, 3, a.size, a.size), device=device).float()
              for _ in range(n)]

    probe = make()
    print(describe(probe))
    needs_geom = probe.needs_geometry
    depths = poses = None
    if needs_geom:
        depths = [torch.rand(1, 1, a.size, a.size, device=device) * 20.0 + 1.0
                  for _ in range(n)]
        poses = []
        for t in range(n):
            p = torch.zeros(1, 7, device=device)
            p[0, 6] = 1.0
            p[0, 0] = 0.15 * t          # forward
            p[0, 1] = 0.02 * t          # slight lateral drift
            poses.append(p)

    def run(model, runner=None):
        out = []
        with torch.no_grad():
            for t in range(n):
                kw = dict(depth=depths[t], pose=poses[t]) if needs_geom else {}
                if runner is None:
                    r = model.forward_frame(frames[t], is_init_frame=(t == 0), **kw)
                else:
                    r = runner.run(frames[t], is_init_frame=(t == 0), **kw)
                out.append(r["logits"].clone())
        return out

    eager = run(probe)

    graph_model = make()
    runner = SAMVTPFullGraphReplayRunner(graph_model, device, enabled=True, capture_warmup=1)
    graphed = run(graph_model, runner)

    print(runner.status_note())
    worst, worst_t = 0.0, -1
    first_replay = runner.min_history_frames + 1
    for t, (e, g) in enumerate(zip(eager, graphed)):
        d = (e - g).abs().max().item()
        if d > worst:
            worst, worst_t = d, t
    print(f"frames compared  : {n} ({first_replay} eager warmup, rest replayed)")
    print(f"max |dlogit|     : {worst:.3e} (frame {worst_t})")
    if not runner._captured:
        print("NOTE: capture never happened, so this compared eager against eager.")
        return 1
    ok = worst <= a.tolerance
    print("PARITY OK" if ok else f"PARITY FAILED (tolerance {a.tolerance:.1e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
