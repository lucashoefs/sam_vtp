#!/usr/bin/env python3
"""Where a SAM-VTP frame's time goes on this machine.

Run it on the deployment target. The FLOP budget is fixed by the architecture
and is the same everywhere, but which part of it dominates is not: on a desktop
GPU the model is launch-bound and CUDA-graph replay is the whole story, while on
an embedded part the attention kernels and memory bandwidth usually are.

Three questions it answers, in order of how much they tend to matter:

  1. Does this build serve SAM-VTP's memory cross-attention with a fused kernel?
     Those layers run one head at head_dim 256 over ~7200 keys. Flash accepts
     that shape; cuDNN's kernel refuses head_dim > 128; the math fallback
     materialises a 1024x7200 score matrix per layer, four times per frame. The
     gap between fused and fallback is close to an order of magnitude, and
     nothing in the model reports which one it got.
  2. What is the per-stage split, eager and under graph replay?
  3. What fraction of the device's arithmetic peak is actually reached?

Timing needs the GPU to itself. A concurrent training job makes every number
here meaningless, so this refuses to run when the GPU is already busy unless
--allow-busy-gpu says otherwise. Where the driver cannot enumerate compute
processes at all -- which is the normal case on Tegra, where nvidia-smi answers
"[N/A]" rather than "none" -- it falls back to timing a fixed kernel and looking
at the spread, since a contended GPU is an uneven one.

    python scripts/profile_device.py --height 512 --width 512
    python scripts/profile_device.py --checkpoint sam_vtp.pt --frames 200
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sam_vtp_core.architecture import SAMVTP, SAMVTPConfig  # noqa: E402
from sam_vtp_core.factory import build_model, describe      # noqa: E402
from sam_vtp_core.fullgraph import SAMVTPFullGraphReplayRunner  # noqa: E402

# Measured with scripts/../ analysis at 512x512, bank full. Attention matmuls
# included -- torch's FlopCounterMode does not attribute them, and they are 36%
# of the total, so a budget taken from it alone understates the model by half.
GFLOP_AT_512 = 115.8


def _sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ""


def gpu_compute_processes():
    """(processes, note). `note` is set when we cannot actually tell.

    On Tegra the GPU is integrated and nvidia-smi does not track compute apps:
    the query returns a row of literal "[N/A]" fields rather than no rows at
    all. Reading that as two live processes is what made this refuse to run on
    the very target it was written for, so a row only counts when its pid field
    is a number.
    """
    raw = _sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader")
    procs = [ln.strip() for ln in raw.splitlines()
             if ln.split(",")[0].strip().isdigit()]
    if procs:
        return procs, ""
    if raw.strip():
        return [], ("nvidia-smi cannot enumerate compute processes on this GPU "
                    "(usual on Tegra)")
    return [], ""


def contention_probe(dev: torch.device, note: str) -> None:
    """Ask the GPU itself whether it is busy, when the driver will not say.

    A contended GPU does not just run slower, it runs *unevenly*: the same
    kernel lands between other people's work and the spread blows up. Timing
    one fixed matmul repeatedly and looking at p90/median detects that without
    needing to know who else is running.
    """
    print(f"\n--- contention probe " + "-" * 51)
    print(f"  {note}, so measuring the spread of a fixed kernel instead.")
    a = torch.randn(1024, 1024, device=dev, dtype=torch.float16)
    b = torch.randn(1024, 1024, device=dev, dtype=torch.float16)
    for _ in range(10):
        a @ b
    samples = sorted(timed(lambda: a @ b, 60, dev))
    med = statistics.median(samples)
    p90 = samples[int(0.90 * (len(samples) - 1))]
    ratio = p90 / med if med > 0 else 1.0
    print(f"  median {med * 1e3:7.1f} us   p90 {p90 * 1e3:7.1f} us   "
          f"p90/median {ratio:.2f}")
    if ratio > 1.5:
        print("  ^ SPREAD IS HIGH: something else is very likely using the GPU."
              "\n    Treat every timing below as contended.")
    else:
        print("  spread is tight; the GPU looks idle apart from this process.")
    del a, b
    torch.cuda.empty_cache()


def report_device(dev: torch.device) -> None:
    print("=" * 72)
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"device      : {p.name}  sm_{p.major}{p.minor}  "
              f"{p.multi_processor_count} SMs  {p.total_memory / 2**30:.1f} GiB")
    print(f"torch       : {torch.__version__}  cuda {torch.version.cuda}")

    # Jetson: the power mode caps clocks and memory bandwidth, and the default
    # mode on a fresh image is not the fastest one. Worth seeing before timing.
    model_name = Path("/proc/device-tree/model")
    if model_name.exists():
        try:
            print(f"board       : {model_name.read_text().strip(chr(0))}")
        except Exception:
            pass
        nvp = _sh("nvpmodel -q 2>/dev/null | head -4")
        if nvp:
            print("nvpmodel    : " + nvp.replace("\n", " | "))
        # jetson_clocks only pins to the ceiling of the *active* mode, so a
        # stack that pins clocks but never sets the mode silently runs at the
        # boot default. Say which of the two situations this is.
        if "SUPER" in nvp.upper() or "MAXN" in nvp.upper():
            print("            ^ top power mode: GPU clock and memory bandwidth are"
                  "\n              uncapped. Nothing to gain here.")
        elif nvp:
            print("            ^ NOT the top power mode. On JetPack 6.2+ the 'Super'"
                  "\n              modes raise GPU clock and memory bandwidth well above"
                  "\n              the launch defaults; jetson_clocks cannot exceed the"
                  "\n              active mode's cap. `sudo nvpmodel -m <id>` first.")
    print("=" * 72)


def check_sdpa(dev: torch.device, feat_tokens: int, mem_tokens: int) -> None:
    """Report, and time, the backends available for the model's own shapes."""
    print("\n--- SDPA backends for SAM-VTP's attention shapes " + "-" * 24)
    if dev.type != "cuda":
        print("  (cuda only)")
        return
    try:
        from torch.backends.cuda import (can_use_flash_attention,
                                         can_use_efficient_attention, SDPAParams)
        from torch.nn.attention import sdpa_kernel, SDPBackend
    except ImportError:
        print("  torch too old to introspect backend selection; skipping")
        return

    shapes = [
        ("memory cross-attn", 1, 1, feat_tokens, mem_tokens, 256),
        ("memory self-attn", 1, 1, feat_tokens, feat_tokens, 256),
    ]
    for name, B, h, N, M, d in shapes:
        q = torch.randn(B, h, N, d, device=dev, dtype=torch.float16)
        k = torch.randn(B, h, M, d, device=dev, dtype=torch.float16)
        v = torch.randn(B, h, M, d, device=dev, dtype=torch.float16)
        gf = 2 * 2 * B * h * N * M * d / 1e9
        try:
            p = SDPAParams(q, k, v, None, 0.0, False, False)
            avail = (f"flash={can_use_flash_attention(p, False)} "
                     f"mem_efficient={can_use_efficient_attention(p, False)}")
        except Exception:
            avail = "(availability query unsupported)"
        print(f"  {name}: q({N},{d}) k({M},{d})   {avail}")

        best = None
        for label, backend in [("math", SDPBackend.MATH),
                               ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
                               ("flash", SDPBackend.FLASH_ATTENTION)]:
            try:
                with sdpa_kernel([backend]):
                    for _ in range(3):
                        F.scaled_dot_product_attention(q, k, v)
                    torch.cuda.synchronize()
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record()
                    for _ in range(20):
                        F.scaled_dot_product_attention(q, k, v)
                    e1.record()
                    torch.cuda.synchronize()
                    ms = e0.elapsed_time(e1) / 20
                print(f"      {label:14s} {ms:8.3f} ms  {gf / ms:6.2f} TFLOP/s")
                best = ms if best is None else min(best, ms)
            except Exception:
                print(f"      {label:14s} unavailable on this build")
        # What the dispatcher picks unaided is what the model actually gets.
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(20):
            F.scaled_dot_product_attention(q, k, v)
        e1.record()
        torch.cuda.synchronize()
        default_ms = e0.elapsed_time(e1) / 20
        verdict = "" if best is None or default_ms <= best * 1.15 else \
            "   <-- SLOWER THAN AN AVAILABLE BACKEND"
        print(f"      {'default':14s} {default_ms:8.3f} ms{verdict}")
        del q, k, v
        torch.cuda.empty_cache()


def timed(fn, n: int, dev: torch.device) -> List[float]:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    out = []
    for _ in range(n):
        if dev.type == "cuda":
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            out.append(e0.elapsed_time(e1))
        else:
            import time
            t = time.perf_counter()
            fn()
            out.append((time.perf_counter() - t) * 1e3)
    return out


def stage_breakdown(model: SAMVTP, img, dep, dev, n: int) -> None:
    """Per-stage cost, each stage synchronised.

    Synchronising after every stage serialises work the GPU would otherwise
    overlap, so these add up to more than the end-to-end frame time. They show
    proportion, not absolute cost.
    """
    core = model.sam2_model
    print("\n--- per-stage, eager, synchronised " + "-" * 37)

    with torch.no_grad():
        image_norm = model.normalize_imagenet(img)
        stages: Dict[str, callable] = {}
        stages["image encoder"] = lambda: core.forward_image(image_norm)

        backbone_out = core.forward_image(image_norm)
        if model.use_geometry:
            from sam_vtp_core.architecture import geometry_channels
            top = backbone_out["backbone_fpn"][-1]
            stages["geometry channels"] = lambda: geometry_channels(
                dep, out_hw=(model.geometry_hw, model.geometry_hw), cam=model.camera)
            geo = geometry_channels(dep, out_hw=(model.geometry_hw, model.geometry_hw),
                                    cam=model.camera)
            stages["geometry encoder"] = lambda: model.geometry_encoder(
                geo.to(top.dtype), top.shape[-2:])
            backbone_out["backbone_fpn"] = list(backbone_out["backbone_fpn"])
            backbone_out["backbone_fpn"][-1] = top + model.geometry_encoder(
                geo.to(top.dtype), top.shape[-2:])

        vf, vp, fs = core.prepare_backbone_features(backbone_out)
        stages["memory attention"] = lambda: core.prepare_memory_conditioned_features(
            frame_idx=model.frame_idx, current_vision_feats=[vf[-1]],
            current_vision_pos_embeds=[vp[-1]], feat_sizes=[fs[-1]],
            output_dict=model.output_dict)

        pix = core.prepare_memory_conditioned_features(
            frame_idx=model.frame_idx, current_vision_feats=[vf[-1]],
            current_vision_pos_embeds=[vp[-1]], feat_sizes=[fs[-1]], output_dict=shadow)
        sparse, dense, pe = model._prompt_tensors(pix, 1)
        hrf = [f.permute(1, 2, 0).view(1, -1, *s) for f, s in zip(vf[:-1], fs[:-1])]
        stages["mask decoder"] = lambda: core.sam_mask_decoder(
            image_embeddings=pix, image_pe=pe, sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense, multimask_output=model.multimask_output,
            repeat_image=False, high_res_features=hrf)

        low_res = core.sam_mask_decoder(
            image_embeddings=pix, image_pe=pe, sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense, multimask_output=model.multimask_output,
            repeat_image=False, high_res_features=hrf)[0]
        H, W = img.shape[-2:]
        stages["upsample to full res"] = lambda: F.interpolate(
            low_res, size=(H, W), mode="bicubic", align_corners=False)
        full = F.interpolate(low_res, size=(H, W), mode="bicubic", align_corners=False)
        if model.refine_head is not None and model.use_refine_head:
            stages["refinement head"] = lambda: model.refine_head(full, image_norm)
        stages["memory encoder"] = lambda: core.encode_new_memory(
            current_vision_feats=vf, feat_sizes=fs, pred_masks_high_res=full,
            object_score_logits=torch.zeros(1, 1, device=full.device))

        results = {}
        for name, fn in stages.items():
            for _ in range(3):
                fn()
            results[name] = statistics.median(timed(fn, n, dev))

    total = sum(results.values())
    for name, ms in sorted(results.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(round(40 * ms / total))
        print(f"  {name:22s} {ms:7.2f} ms  {100 * ms / total:5.1f}%  {bar}")
    print(f"  {'(serialised sum)':22s} {total:7.2f} ms")


def replay_cost_by_component(model, dev, mk, frames: int) -> None:
    """What each optional stage costs in the regime the robot actually runs in.

    The hook breakdown below can only run eagerly -- hooks do not fire inside a
    captured graph -- so it measures a regime the deployment is not in, and
    over-attributes to whatever launches the most kernels. Switching a component
    off, re-capturing, and taking the difference measures the thing that matters,
    and it uses the same switches the ablation study does, so a stage that looks
    expensive here is a stage worth optimising rather than an artefact of how the
    measurement was taken.

    The differences do not have to sum to the total: turning one stage off can
    let the rest overlap differently.
    """
    print("\n--- cost under graph replay, by switching stages off " + "-" * 19)
    have = {"refine": model.refine_head is not None and model.use_refine_head,
            "geometry": model.geometry_encoder is not None and model.use_geometry,
            }

    def measure(label: str, **switches) -> Optional[float]:
        previous = {k: getattr(model, {"refine": "use_refine_head",
                                       "geometry": "use_geometry"}[k])
                    for k in switches}
        try:
            model.set_components(**switches)
            model.reset_memory()
            runner = SAMVTPFullGraphReplayRunner(model, dev, enabled=True,
                                                 capture_warmup=2)
            for t in range(max(20, runner.min_history_frames + 3)):
                img, dep = mk(t)
                runner.run(img, is_init_frame=(t == 0), depth=dep)
            if not runner._captured:
                print(f"  {label:24s} capture did not engage; skipped")
                return None
            img, dep = mk(99)
            ms = statistics.median(timed(
                lambda: runner.run(img, is_init_frame=False, depth=dep),
                frames, dev))
            return ms
        finally:
            model.set_components(**previous)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        base = measure("baseline", **{})
        if base is None:
            return
        print(f"  {'all stages on':24s} {base:7.2f} ms")
        for name, on in have.items():
            if not on:
                continue
            key = {"refine": "refine", "geometry": "geometry"}[name]
            off = measure(f"{name} off", **{key: False})
            if off is None:
                continue
            print(f"  {'without ' + name:24s} {off:7.2f} ms   "
                  f"{name} costs {base - off:6.2f} ms "
                  f"({100 * (base - off) / base:4.1f}% of the frame)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None,
                    help="optional; random weights time identically")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--amp", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--no-graph", action="store_true", help="skip the graph-replay pass")
    ap.add_argument("--skip-stages", action="store_true")
    ap.add_argument("--cudnn-benchmark", action="store_true",
                    help="let cuDNN pick convolution algorithms by measurement "
                         "(what tune_for_inference does in deployment)")
    ap.add_argument("--channels-last", action="store_true",
                    help="run the whole model in NHWC. Narrow convolutions are "
                         "much better served by it; attention is indifferent")
    ap.add_argument("--skip-ablate", action="store_true",
                    help="skip the switch-a-stage-off pass (it re-captures the "
                         "graph once per stage, so it is the slow part)")
    ap.add_argument("--allow-busy-gpu", action="store_true",
                    help="time anyway with another process on the GPU (numbers will be junk)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report_device(dev)

    probe_note = ""
    if dev.type == "cuda" and not args.allow_busy_gpu:
        busy, probe_note = gpu_compute_processes()
        if busy:
            print("\nREFUSING TO TIME: the GPU already has compute processes on it:\n  "
                  + "\n  ".join(busy)
                  + "\nContention makes every number below meaningless. Wait for it to"
                    "\nfinish, or pass --allow-busy-gpu if you only want the SDPA check.")
            return 1

    if args.checkpoint:
        model = build_model(args.checkpoint, device=dev)
    else:
        model = SAMVTP(SAMVTPConfig()).to(dev).eval()
        print("\n(random weights: timing is weight-independent)")
    print(describe(model))

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print("cudnn.benchmark = True")
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("model converted to channels_last (NHWC)")

    stride = model.sam2_model.backbone_stride
    feat_tokens = (args.height // stride) * (args.width // stride)
    mem_tokens = feat_tokens * model.num_maskmem + 4 * model.num_maskmem
    check_sdpa(dev, feat_tokens, mem_tokens)

    if args.allow_busy_gpu:
        print("\n--allow-busy-gpu: timings below may be contended and not comparable.")
    elif probe_note and dev.type == "cuda":
        contention_probe(dev, probe_note)

    amp = torch.float16 if args.amp == "fp16" else (
        torch.bfloat16 if args.amp == "bf16" else torch.float32)
    ctx = (torch.autocast("cuda", dtype=amp)
           if dev.type == "cuda" and args.amp != "fp32" else torch.no_grad())

    def mk(t):
        img = torch.rand(1, 3, args.height, args.width, device=dev) * 255
        dep = torch.rand(1, 1, args.height, args.width, device=dev) * 20 + 1.0
        return img, dep

    print("\n--- end to end " + "-" * 57)
    with torch.no_grad(), ctx:
        model.reset_memory()
        for t in range(20):                      # fill bank + warm caches/algos
            img, dep = mk(t)
            model.forward_frame(img, is_init_frame=(t == 0), depth=dep)
        img, dep = mk(20)
        eager = timed(lambda: model.forward_frame(img, depth=dep),
                      args.frames, dev)
        med = statistics.median(eager)
        scale = (args.height * args.width) / (512 * 512)
        gf = GFLOP_AT_512 * scale
        print(f"  eager           {med:7.2f} ms  {1000 / med:6.2f} FPS  "
              f"{gf / med:6.2f} TFLOP/s effective")

        if not args.no_graph and dev.type == "cuda":
            model.reset_memory()
            runner = SAMVTPFullGraphReplayRunner(model, dev, enabled=True, capture_warmup=2)
            for t in range(max(20, runner.min_history_frames + 3)):
                img, dep = mk(t)
                runner.run(img, is_init_frame=(t == 0), depth=dep)
            print(f"  graph status    {runner.status_note()}")
            img, dep = mk(99)
            g = timed(lambda: runner.run(img, is_init_frame=False, depth=dep),
                      args.frames, dev)
            gmed = statistics.median(g)
            print(f"  graph replay    {gmed:7.2f} ms  {1000 / gmed:6.2f} FPS  "
                  f"{gf / gmed:6.2f} TFLOP/s effective   ({med / gmed:.2f}x over eager)")

    if not args.no_graph and dev.type == "cuda" and not args.skip_ablate:
        replay_cost_by_component(model, dev, mk, max(20, args.frames // 2))

    if not args.skip_stages:
        model.reset_memory()
        with torch.no_grad(), ctx:
            for t in range(20):
                img, dep = mk(t)
                model.forward_frame(img, is_init_frame=(t == 0), depth=dep)
            img, dep = mk(20)
            stage_breakdown(model, img, dep, dev, max(20, args.frames // 4))

    print(f"\nFLOP budget at {args.height}x{args.width}: {gf:.1f} GFLOP/frame "
          f"(53% image encoder, 42% memory attention, 5% everything else)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
