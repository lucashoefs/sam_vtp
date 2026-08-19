#!/usr/bin/env python3
"""Check that CUDA-graph replay agrees with the eager forward pass.

`fullgraph.py` is a second implementation of the forward pass, written over
tensorised state so it can be captured. Two implementations drift. This runs the
same sequence through both and reports the largest disagreement, so a change to
`architecture.py` that the graph path did not follow shows up as a number rather
than as quietly different results whenever the speed flag is on.

Both are run in the model's own configuration, so a geometry checkpoint tests
the geometry encoder inside the graph. `--rgb-only` forces the
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
                    help="force use_geometry off")
    ap.add_argument("--no-memory", action="store_true",
                    help="force memory off, exercising the frame-wise capture "
                         "path. That path has no recurrent state, so it is "
                         "captured directly rather than after the bank fills")
    ap.add_argument("--channels-last", action="store_true",
                    help="run both models in NHWC, checking that graph capture "
                         "still agrees with eager in that layout. This does NOT "
                         "measure what the layout does to the mask -- for that, "
                         "evaluate NCHW against NHWC on real footage")
    ap.add_argument("--memory-write-every", type=int, default=1,
                    help="write to the bank every k-th frame. >1 exercises the "
                         "second captured graph, and checks that the cadence "
                         "stays in phase across the eager-to-replay handover")
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
        switches = {}
        if a.rgb_only:
            switches["geometry"] = False
        if a.no_memory:
            switches["memory"] = False
        if switches:
            m.set_components(**switches)
        m.memory_write_every = max(1, int(a.memory_write_every))
        if a.channels_last:
            m.to(memory_format=torch.channels_last)
        return m

    torch.manual_seed(0)
    every = max(1, int(a.memory_write_every))
    # The bank fills `every` times slower, and capture waits for a full bank, so
    # a run long enough at every=1 replays nothing at every=3.
    n = a.frames if (every == 1 or a.no_memory) else max(a.frames, 24 * every)
    frames = [torch.randint(0, 256, (1, 3, a.size, a.size), device=device).float()
              for _ in range(n)]

    probe = make()
    print(describe(probe))
    needs_geom = probe.needs_geometry
    depths = None
    if needs_geom:
        depths = [torch.rand(1, 1, a.size, a.size, device=device) * 20.0 + 1.0
                  for _ in range(n)]

    def run(model, runner=None):
        out = []
        with torch.no_grad():
            for t in range(n):
                kw = dict(depth=depths[t]) if needs_geom else {}
                if runner is None:
                    r = model.forward_frame(frames[t], is_init_frame=(t == 0), **kw)
                else:
                    r = runner.run(frames[t], is_init_frame=(t == 0), **kw)
                out.append(r["logits"].clone())
        return out

    eager = run(probe)

    graph_model = make()
    runner = SAMVTPFullGraphReplayRunner(graph_model, device, enabled=True,
                                         capture_warmup=1,
                                         capture_nowrite=every > 1 and not a.no_memory)
    graphed = run(graph_model, runner)

    print(runner.status_note())
    worst, worst_t = 0.0, -1
    first_replay = runner.min_history_frames + 1
    for t, (e, g) in enumerate(zip(eager, graphed)):
        d = (e - g).abs().max().item()
        if d > worst:
            worst, worst_t = d, t
    print(f"frames compared  : {n} ({first_replay} eager warmup, rest replayed)")
    where = "no frame disagreed" if worst_t < 0 else f"frame {worst_t}"
    print(f"max |dlogit|     : {worst:.3e} ({where})")
    if not runner._captured:
        print("NOTE: capture never happened, so this compared eager against eager.")
        return 1
    if every > 1:
        # Divergent cadence is the failure this flag exists to catch, and it
        # would otherwise look like a small numeric drift rather than the two
        # models holding different memory.
        print(f"write cadence    : every {every} frames; "
              f"eager wrote {probe.frame_idx}, graph wrote {graph_model.frame_idx} "
              f"(+{runner._state.step_idx.item() - graph_model.frame_idx:.0f} "
              f"inside the graph)")
        if runner._graph_nowrite is None:
            print("NOTE: no non-writing graph was captured; the cadence was not tested.")
            return 1
    ok = worst <= a.tolerance
    print("PARITY OK" if ok else f"PARITY FAILED (tolerance {a.tolerance:.1e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
