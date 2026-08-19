#!/usr/bin/env python3
"""What memory and geometry cost per frame, in the 2x2 the ablation reports.

This exists to put a latency column next to Table II, so each accuracy row has
its matched cost and "is the gain worth it" becomes answerable per row rather
than in the abstract.

Run it on the deployment target. No checkpoint is needed: latency is fixed by
shapes and control flow, not by weight values, so a randomly initialised model
times identically to a trained one. That also makes this runnable on a robot
that does not yet have the checkpoint.

Every arm is timed in both regimes -- eager, and under CUDA-graph replay -- and
the two are reported side by side. Reporting only one is what makes this
measurement easy to get wrong: replay is worth a large constant on a
launch-bound device, so timing the full model under replay against a frame-wise
arm running eagerly charges that entire constant to the frame-wise arm and makes
the memory pathway look far cheaper than it is. `fullgraph.py` used to refuse to
capture the frame-wise configuration at all, which left that mismatched
comparison as the only one available; it now captures both, so the deployed-
regime column is a like-for-like comparison and is the one to quote.

    python scripts/bench_components.py                     # 512x512, fp16
    python scripts/bench_components.py --frames 200
"""
from __future__ import annotations

import argparse
import gc
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


def _stats(samples: List[float]) -> Dict[str, float]:
    samples = sorted(samples)
    return {"median": statistics.median(samples),
            "p10": samples[int(0.10 * (len(samples) - 1))],
            "p90": samples[int(0.90 * (len(samples) - 1))]}


def bench_arm(model: SAMVTP, dev: torch.device, mk, *, memory: bool, geometry: bool,
              warmup: int, frames: int, amp: Optional[torch.dtype],
              want_graph: bool) -> Dict[str, object]:
    """Median ms/frame for one configuration, eager and under graph replay.

    Memory-on arms need the bank filled before the steady-state cost is real --
    an unfilled bank means fewer keys in memory attention and an optimistic
    number -- so warmup is at least num_maskmem frames regardless of --warmup.
    """
    model.set_components(memory=memory, geometry=geometry)
    ctx = lambda: (torch.autocast("cuda", dtype=amp)  # noqa: E731
                   if dev.type == "cuda" and amp is not None else torch.no_grad())
    dep_or_none = lambda d: d if geometry else None  # noqa: E731
    need = max(warmup, int(model.num_maskmem) + 3) if memory else warmup

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    model.reset_memory()
    with torch.no_grad(), ctx():
        for t in range(need):
            img, dep = mk(t)
            model.forward_frame(img, is_init_frame=(t == 0), depth=dep_or_none(dep))
        img, dep = mk(need)
        eager = _stats(timed(
            lambda: model.forward_frame(img, depth=dep_or_none(dep)), frames, dev))

    out: Dict[str, object] = {"eager": eager, "replay": None, "status": "not attempted",
                              "peak_mb": 0.0}
    if not (want_graph and dev.type == "cuda"):
        return out

    model.reset_memory()
    runner = SAMVTPFullGraphReplayRunner(model, dev, enabled=True, capture_warmup=2)
    with torch.no_grad(), ctx():
        for t in range(max(need, runner.min_history_frames + 3)):
            img, dep = mk(t)
            runner.run(img, is_init_frame=(t == 0), depth=dep_or_none(dep))
        out["status"] = runner.status_note()
        if runner._captured:
            img, dep = mk(99)
            out["replay"] = _stats(timed(
                lambda: runner.run(img, is_init_frame=False, depth=dep_or_none(dep)),
                frames, dev))
    out["peak_mb"] = torch.cuda.max_memory_reserved(dev) / 2**20

    # Each arm captures its own graphs into its own pool. On an 8 GB Orin four
    # live pools is not free, so drop this one before the next arm builds its.
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    return out


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
                    help="eager only; skips the deployed-regime column")
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

    results = []
    for label, mem, geo in ARMS:
        r = bench_arm(model, dev, mk, memory=mem, geometry=geo, warmup=args.warmup,
                      frames=args.frames, amp=amp, want_graph=not args.no_graph)
        results.append((label, mem, geo, r))
        print(f"  [{label.strip():22s}] {r['status']}")

    scale = (args.height * args.width) / (512 * 512)
    gf = GFLOP_AT_512 * scale
    captured = [r for _, _, _, r in results if r["replay"] is not None]
    all_captured = len(captured) == len(results)

    print(f"\n--- per-frame cost, {args.height}x{args.width}, {args.amp} " + "-" * 30)
    print(f"  {'configuration':22s} {'eager ms':>9s} {'replay ms':>10s} "
          f"{'FPS':>7s} {'vs full':>9s} {'peak MB':>9s}")
    full_key = "replay" if all_captured else "eager"
    full = results[-1][3][full_key]["median"]
    for label, mem, geo, r in results:
        rep = r["replay"]
        best = (rep or r["eager"])["median"]
        rep_s = f"{rep['median']:10.2f}" if rep else f"{'--':>10s}"
        rel = "--" if best == full else f"{100 * (best - full) / full:+.1f}%"
        print(f"  {label:22s} {r['eager']['median']:9.2f} {rep_s} "
              f"{1000 / best:7.2f} {rel:>9s} {r['peak_mb']:9.0f}")
    print(f"  FPS and 'vs full' use the {full_key} column"
          + ("" if all_captured else "  (not every arm captured -- see status above)"))

    for regime in (["eager", "replay"] if all_captured else ["eager"]):
        by = {(m, g): r[regime]["median"] for _, m, g, r in results}
        print(f"\n--- what each component costs, {regime} " + "-" * (41 - len(regime)))
        for name, on, off, cond in (
                ("memory",   (True, False), (False, False), "geometry off"),
                ("memory",   (True, True),  (False, True),  "geometry on"),
                ("geometry", (False, True), (False, False), "memory off"),
                ("geometry", (True, True),  (True, False),  "memory on")):
            d = by[on] - by[off]
            print(f"  {name:9s} with {cond:13s} {d:+7.2f} ms  "
                  f"({100 * d / by[on]:5.1f}% of that frame)")

    if all_captured:
        print("\n  Both regimes are like-for-like: every arm above was captured, so the"
              "\n  replay column is a fair deployed-regime comparison. Quote that one.")
    else:
        print("\n  Some arm did not capture, so the replay column is NOT like-for-like."
              "\n  Do not subtract a replayed number from an eager one; use the eager"
              "\n  column, which every arm can run in.")

    print(f"\nFLOP budget at {args.height}x{args.width}: {gf:.1f} GFLOP/frame "
          f"(53% image encoder, 42% memory attention, 5% everything else)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
