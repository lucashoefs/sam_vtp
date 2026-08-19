#!/usr/bin/env python3
"""What memory and geometry cost per frame, in the 2x2 the ablation reports.

This exists to put a latency column next to Table II, so each accuracy row has
its matched cost and "is the gain worth it" becomes answerable per row rather
than in the abstract.

Run it on the deployment target. No checkpoint is needed: latency is fixed by
shapes and control flow, not by weight values, so a randomly initialised model
times identically to a trained one. That also makes this runnable on a robot
that does not yet have the checkpoint.

WHY EVERY ARM IS TIMED EAGER
----------------------------
The obvious approach -- time each configuration the way it would be deployed --
does not work here, and getting it wrong biases the result in the flattering
direction. `SAMVTPFullGraphReplayRunner` refuses to capture when memory is off:

    self._framewise = not bool(getattr(model, "use_memory", True))
    if self.enabled and self._framewise:
        self.enabled = False

so a "deploy each arm normally" sweep would time SAM-VTP under CUDA-graph replay
and SAM-TP eagerly. Replay is worth a large constant on any launch-bound device,
all of which would be charged to the frame-wise arm -- making the memory pathway
look cheaper than it is, or free. That is exactly the direction that flatters the
paper's argument, so it is the one to refuse.

Eager for every arm is the only regime all four configurations can actually run
in, so it is what the comparison uses. It is not the regime the robot deploys
in; the deployed full-model number is printed separately below, clearly marked,
and the two must not be subtracted from each other.

The eager numbers still answer the question the paper asks -- what fraction of
frame compute each component is -- and on an embedded part they answer it well:
Orin-class hardware is attention- and bandwidth-bound where a desktop GPU is
launch-bound, so eager and replay sit much closer together there.

    python scripts/bench_components.py                     # 512x512, fp16
    python scripts/bench_components.py --frames 200 --no-graph
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE))

from sam_vtp_core.architecture import SAMVTP, SAMVTPConfig      # noqa: E402
from sam_vtp_core.factory import describe                        # noqa: E402
from sam_vtp_core.fullgraph import SAMVTPFullGraphReplayRunner   # noqa: E402
from profile_device import (                                     # noqa: E402
    GFLOP_AT_512, contention_probe, gpu_compute_processes, report_device, timed,
)

# (label, memory, geometry) in the order Table II lists them.
ARMS: Tuple[Tuple[str, bool, bool], ...] = (
    ("SAM-TP (frame-wise)",   False, False),
    ("  + geometry",          False, True),
    ("SAM-VTP (memory)",      True,  False),
    ("  + geometry (full)",   True,  True),
)


def bench_arm(model: SAMVTP, dev: torch.device, mk, *, memory: bool, geometry: bool,
              warmup: int, frames: int, amp: Optional[torch.dtype]) -> Dict[str, float]:
    """Median ms/frame for one configuration, timed eagerly.

    Memory-on arms need the bank filled before the steady-state cost is real --
    an unfilled bank means fewer keys in memory attention and an optimistic
    number -- so warmup is at least num_maskmem frames regardless of --warmup.
    """
    model.set_components(memory=memory, geometry=geometry)
    model.reset_memory()

    ctx = (torch.autocast("cuda", dtype=amp)
           if dev.type == "cuda" and amp is not None else torch.no_grad())
    need = max(warmup, int(model.num_maskmem) + 3) if memory else warmup

    with torch.no_grad(), ctx:
        for t in range(need):
            img, dep = mk(t)
            model.forward_frame(img, is_init_frame=(t == 0),
                                depth=dep if geometry else None)
        img, dep = mk(need)
        samples = timed(
            lambda: model.forward_frame(img, depth=dep if geometry else None),
            frames, dev)

    samples.sort()
    return {"median": statistics.median(samples),
            "p10": samples[int(0.10 * (len(samples) - 1))],
            "p90": samples[int(0.90 * (len(samples) - 1))]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--frames", type=int, default=100, help="timed frames per arm")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--amp", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--channels-last", dest="channels_last", action="store_true",
                    default=True,
                    help="NHWC, as tune_for_inference does in deployment (default)")
    ap.add_argument("--no-channels-last", dest="channels_last", action="store_false")
    ap.add_argument("--no-graph", action="store_true",
                    help="skip the deployed full-model graph-replay reference")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report_device(dev)

    probe_note = ""
    if dev.type == "cuda" and not args.allow_busy_gpu:
        busy, probe_note = gpu_compute_processes()
        if busy:
            print("\nREFUSING TO TIME: the GPU already has compute processes on it:\n  "
                  + "\n  ".join(busy)
                  + "\nContention makes every number below meaningless.")
            return 1

    model = SAMVTP(SAMVTPConfig()).to(dev).eval()
    print("\n(random weights: timing is weight-independent, no checkpoint needed)")
    print(describe(model))
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("channels_last (NHWC): on")

    if probe_note and dev.type == "cuda" and not args.allow_busy_gpu:
        contention_probe(dev, probe_note)

    amp = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[args.amp]

    # One fixed input for every arm, so no arm gets a different cache state.
    gen = torch.Generator(device="cpu").manual_seed(0)
    img0 = (torch.rand(1, 3, args.height, args.width, generator=gen) * 255).to(dev)
    dep0 = (torch.rand(1, 1, args.height, args.width, generator=gen) * 20 + 1.0).to(dev)
    if args.channels_last:
        img0 = img0.to(memory_format=torch.channels_last)
    mk = lambda t: (img0, dep0)  # noqa: E731

    results: List[Tuple[str, bool, bool, Dict[str, float]]] = []
    for label, mem, geo in ARMS:
        r = bench_arm(model, dev, mk, memory=mem, geometry=geo,
                      warmup=args.warmup, frames=args.frames, amp=amp)
        results.append((label, mem, geo, r))

    full = results[-1][3]["median"]
    scale = (args.height * args.width) / (512 * 512)
    gf = GFLOP_AT_512 * scale

    print(f"\n--- per-frame cost, eager, {args.height}x{args.width}, {args.amp} " + "-" * 18)
    print(f"  {'configuration':22s} {'ms':>8s} {'FPS':>7s} {'p10-p90':>14s} {'vs full':>9s}")
    for label, mem, geo, r in results:
        rel = "--" if r["median"] == full else f"{100 * (r['median'] - full) / full:+.1f}%"
        print(f"  {label:22s} {r['median']:8.2f} {1000 / r['median']:7.2f} "
              f"{r['p10']:6.2f}-{r['p90']:6.2f} {rel:>9s}")

    by = {(m, g): r["median"] for _, m, g, r in results}
    print("\n--- what each component costs " + "-" * 42)
    for name, on, off, cond in (
            ("memory",   (True, False), (False, False), "geometry off"),
            ("memory",   (True, True),  (False, True),  "geometry on"),
            ("geometry", (False, True), (False, False), "memory off"),
            ("geometry", (True, True),  (True, False),  "memory on")):
        d = by[on] - by[off]
        print(f"  {name:9s} with {cond:13s} {d:+7.2f} ms  "
              f"({100 * d / by[on]:5.1f}% of that frame)")

    if not args.no_graph and dev.type == "cuda":
        model.set_components(memory=True, geometry=True)
        model.reset_memory()
        runner = SAMVTPFullGraphReplayRunner(model, dev, enabled=True, capture_warmup=2)
        with torch.no_grad(), (torch.autocast("cuda", dtype=amp) if amp else torch.no_grad()):
            for t in range(max(args.warmup, runner.min_history_frames + 3)):
                i, d = mk(t)
                runner.run(i, is_init_frame=(t == 0), depth=d)
            i, d = mk(99)
            g = statistics.median(timed(
                lambda: runner.run(i, is_init_frame=False, depth=d), args.frames, dev))
        print("\n--- deployed full model, graph replay " + "-" * 34)
        print(f"  graph status    {runner.status_note()}")
        print(f"  full + replay   {g:8.2f} ms  {1000 / g:6.2f} FPS  "
              f"({full / g:.2f}x over its own eager number)")
        print("  NOT comparable with the eager arms above -- the frame-wise arms")
        print("  cannot be graph-captured, so do not subtract across the two blocks.")

    print(f"\nFLOP budget at {args.height}x{args.width}: {gf:.1f} GFLOP/frame "
          f"(53% image encoder, 42% memory attention, 5% everything else)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
