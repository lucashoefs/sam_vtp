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
--allow-busy-gpu says otherwise.

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
        print("            (jetson_clocks pins clocks to the cap for the active mode;"
              "\n             JetPack 6.2 'Super' modes raise both clock and memory"
              "\n             bandwidth substantially over the launch defaults)")
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


def stage_breakdown(model: SAMVTP, img, dep, pose, dev, n: int) -> None:
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
        if model.warp_memory:
            stages["memory warp"] = lambda: model._warped_output_dict(dep, pose, fs[-1])
        shadow = (model._warped_output_dict(dep, pose, fs[-1])
                  if model.warp_memory else model.output_dict)
        stages["memory attention"] = lambda: core.prepare_memory_conditioned_features(
            frame_idx=model.frame_idx, current_vision_feats=[vf[-1]],
            current_vision_pos_embeds=[vp[-1]], feat_sizes=[fs[-1]], output_dict=shadow)

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
    ap.add_argument("--allow-busy-gpu", action="store_true",
                    help="time anyway with another process on the GPU (numbers will be junk)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report_device(dev)

    if dev.type == "cuda" and not args.allow_busy_gpu:
        busy = _sh("nvidia-smi --query-compute-apps=pid,used_memory "
                   "--format=csv,noheader")
        if busy:
            print(f"\nREFUSING TO TIME: the GPU already has compute processes on it:\n  "
                  + busy.replace("\n", "\n  ")
                  + "\nContention makes every number below meaningless. Wait for it to"
                    "\nfinish, or pass --allow-busy-gpu if you only want the SDPA check.")
            return 1

    if args.checkpoint:
        model = build_model(args.checkpoint, device=dev)
    else:
        model = SAMVTP(SAMVTPConfig()).to(dev).eval()
        print("\n(random weights: timing is weight-independent)")
    print(describe(model))

    stride = model.sam2_model.backbone_stride
    feat_tokens = (args.height // stride) * (args.width // stride)
    mem_tokens = feat_tokens * model.num_maskmem + 4 * model.num_maskmem
    check_sdpa(dev, feat_tokens, mem_tokens)

    if dev.type == "cuda" and not args.allow_busy_gpu:
        pass
    elif args.allow_busy_gpu:
        print("\n--allow-busy-gpu: timings below are contended and not comparable.")

    amp = torch.float16 if args.amp == "fp16" else (
        torch.bfloat16 if args.amp == "bf16" else torch.float32)
    ctx = (torch.autocast("cuda", dtype=amp)
           if dev.type == "cuda" and args.amp != "fp32" else torch.no_grad())

    def mk(t):
        img = torch.rand(1, 3, args.height, args.width, device=dev) * 255
        dep = torch.rand(1, 1, args.height, args.width, device=dev) * 20 + 1.0
        pose = torch.zeros(1, 7, device=dev)
        pose[0, 6] = 1.0
        pose[0, 0] = 0.2 * t
        return img, dep, pose

    print("\n--- end to end " + "-" * 57)
    with torch.no_grad(), ctx:
        model.reset_memory()
        for t in range(20):                      # fill bank + warm caches/algos
            img, dep, pose = mk(t)
            model.forward_frame(img, is_init_frame=(t == 0), depth=dep, pose=pose)
        img, dep, pose = mk(20)
        eager = timed(lambda: model.forward_frame(img, depth=dep, pose=pose),
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
                img, dep, pose = mk(t)
                runner.run(img, is_init_frame=(t == 0), depth=dep, pose=pose)
            print(f"  graph status    {runner.status_note()}")
            img, dep, pose = mk(99)
            g = timed(lambda: runner.run(img, is_init_frame=False, depth=dep, pose=pose),
                      args.frames, dev)
            gmed = statistics.median(g)
            print(f"  graph replay    {gmed:7.2f} ms  {1000 / gmed:6.2f} FPS  "
                  f"{gf / gmed:6.2f} TFLOP/s effective   ({med / gmed:.2f}x over eager)")

    if not args.skip_stages:
        model.reset_memory()
        with torch.no_grad(), ctx:
            for t in range(20):
                img, dep, pose = mk(t)
                model.forward_frame(img, is_init_frame=(t == 0), depth=dep, pose=pose)
            img, dep, pose = mk(20)
            stage_breakdown(model, img, dep, pose, dev, max(20, args.frames // 4))

    print(f"\nFLOP budget at {args.height}x{args.width}: {gf:.1f} GFLOP/frame "
          f"(53% image encoder, 42% memory attention, 5% everything else)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
