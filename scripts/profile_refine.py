#!/usr/bin/env python3
"""Why the refinement head costs 22% of the frame for 0.4% of the arithmetic.

Measured on an Orin Nano Super, the head is ~13.8 ms of a 61.5 ms frame while
being ~0.46 GFLOP -- about 0.2% of the device's peak, where the rest of the model
reaches 13-16%. At 60-70 MB of traffic it should be under a millisecond, so this
is kernel shape, not bandwidth, and it should be fixable without touching the
architecture or retraining.

Three candidates, all visible in the code and none of them conclusive on their
own:

  * NCHW with 4 and 8 channels. cuDNN's fp16 tensor-core kernels want channel
    counts that tile; 4 input channels cannot use them at all. `channels_last`
    is the usual fix for narrow convolutions.
  * Dtype ping-pong. `normalize_imagenet` is elementwise, so autocast leaves
    `image_norm` in fp32 while the decoder's logits are fp16; `cat` promotes the
    pair back to fp32, the convolutions cast down, and GroupNorm -- on autocast's
    fp32 list -- casts up again.
  * `GroupNorm(1, 8)` is a whole-tensor reduction over 2.1M elements, twice.

So this times the head as the model actually calls it, times each layer inside
it, and then A/B's the variants. Variants that change layout or dtype are NOT
bit-exact, so each reports its own worst disagreement against the baseline -- the
question is whether the difference is smaller than the head's own residual, not
whether it is zero.

    python scripts/profile_refine.py
    python scripts/profile_refine.py --size 384 --frames 200
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam_vtp_core.architecture import RefinementHead  # noqa: E402


def timed(fn, n: int) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        samples.append(e0.elapsed_time(e1))
    return statistics.median(samples)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--channels", type=int, default=8)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required")
        return 77
    dev = torch.device("cuda")
    S = a.size
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | {S}x{S}\n")

    # Seed BEFORE the head is built. Unseeded weights made `max|diff|` and the
    # residual scale wander between runs, which is exactly the pair of numbers
    # this script exists to compare.
    torch.manual_seed(0)
    head = RefinementHead(channels=a.channels).to(dev).eval()
    # Zero-initialised at the output by design, so a trained head is the harder
    # case numerically; give it real weights or every variant agrees trivially.
    with torch.no_grad():
        torch.nn.init.normal_(head.out.weight, std=0.05)
        torch.nn.init.normal_(head.out.bias, std=0.05)

    # Exactly the dtypes forward_frame produces under autocast: image_norm stays
    # fp32 because normalisation is elementwise, logits are fp16 from the decoder.
    image_f32 = torch.randn(1, 3, S, S, device=dev, dtype=torch.float32)
    logits_f16 = torch.randn(1, 1, S, S, device=dev, dtype=torch.float16)

    amp = torch.autocast("cuda", dtype=torch.float16)

    def baseline():
        with torch.no_grad(), amp:
            return head(logits_f16, image_f32)

    ref = baseline().float()
    base_ms = timed(baseline, a.frames)
    gf = (2 * S * S * (a.channels * 4 * 9 + a.channels * a.channels * 9
                       + a.channels)) / 1e9
    print(f"baseline (as the model calls it)   {base_ms:7.3f} ms   "
          f"{gf / base_ms:6.3f} TFLOP/s")
    print(f"  input dtypes: image {image_f32.dtype}, logits {logits_f16.dtype}"
          f"  ->  cat promotes to "
          f"{torch.cat([image_f32, logits_f16.float()], 1).dtype}\n")

    # -- where does it go, layer by layer ------------------------------------
    print("--- inside the head, layer by layer " + "-" * 32)
    with torch.no_grad(), amp:
        x = torch.cat([image_f32, logits_f16], dim=1)
        stages = [("cat([rgb, logits])",
                   lambda: torch.cat([image_f32, logits_f16], dim=1))]
        cur = x
        for i, layer in enumerate(head.stem):
            name = f"stem[{i}] {type(layer).__name__}"
            stages.append((name, (lambda m, t: (lambda: m(t)))(layer, cur)))
            cur = layer(cur)
        stages.append(("out Conv2d(->1)", (lambda t: (lambda: head.out(t)))(cur)))

        total = 0.0
        for name, fn in stages:
            ms = timed(fn, a.frames)
            total += ms
            print(f"  {name:28s} {ms:7.3f} ms")
        print(f"  {'(serialised sum)':28s} {total:7.3f} ms")

    # -- variants ------------------------------------------------------------
    print("\n--- variants " + "-" * 54)
    print(f"  {'variant':34s} {'ms':>8} {'vs base':>9}  {'max|diff|':>10}")
    print(f"  {'baseline':34s} {base_ms:8.3f} {'1.00x':>9}  {'-':>10}")

    def report(label: str, fn) -> None:
        try:
            out = fn().float()
            ms = timed(fn, a.frames)
            d = (out - ref).abs().max().item()
            print(f"  {label:34s} {ms:8.3f} {base_ms / ms:8.2f}x  {d:10.3e}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:34s} failed: {type(exc).__name__}: {exc}")

    # 1. kill the fp32 promotion: match dtypes before the cat.
    # Measured a dead end -- autocast already casts for the convolution, so
    # moving the cast earlier is bit-identical and just as slow. Kept so a
    # future reader does not re-derive it.
    image_f16 = image_f32.half()

    def half_input():
        with torch.no_grad(), amp:
            return head(logits_f16, image_f16)
    report("fp16 image (no cat promotion)", half_input)

    # 2. let cuDNN choose its algorithm by measurement rather than by heuristic.
    # These channel counts are exactly the case its heuristics are worst at, and
    # shapes here are fixed for the life of the process.
    torch.backends.cudnn.benchmark = True
    report("cudnn.benchmark", baseline)

    # 3. channels_last: C=4 and C=8 become the contiguous axis.
    #
    # The conversion of the two inputs is INSIDE the timed region on purpose.
    # The model hands this head NCHW tensors, so a deployment pays that
    # transpose every frame; pre-converting it outside the loop would report a
    # speedup the robot never sees.
    head_cl = RefinementHead(channels=a.channels).to(dev).eval()
    head_cl.load_state_dict(head.state_dict())
    head_cl = head_cl.to(memory_format=torch.channels_last)

    def chlast():
        with torch.no_grad(), amp:
            return head_cl(logits_f16.contiguous(memory_format=torch.channels_last),
                           image_f32.contiguous(memory_format=torch.channels_last))
    report("channels_last (+ conversion)", chlast)

    log_pre = logits_f16.contiguous(memory_format=torch.channels_last)
    img_pre = image_f32.contiguous(memory_format=torch.channels_last)

    def chlast_pre():
        with torch.no_grad(), amp:
            return head_cl(log_pre, img_pre)
    report("channels_last (already NHWC)", chlast_pre)

    torch.backends.cudnn.benchmark = False
    report("channels_last, no cudnn.benchmark", chlast)
    torch.backends.cudnn.benchmark = True

    # 4. pad the 4-channel input to 8. cuDNN's fp16 tensor-core kernels want a
    # channel count that tiles by 8, and 4 cannot use them at all. The added
    # taps multiply by zero, so this is mathematically the same convolution;
    # measured at one fp32 ulp (2.4e-07) on CPU, since four extra zero terms
    # still change the accumulation order.
    with torch.no_grad():
        head_pad = RefinementHead(channels=a.channels).to(dev).eval()
        head_pad.load_state_dict(head.state_dict())
        c0 = head_pad.stem[0]
        wide = torch.nn.Conv2d(8, a.channels, 3, padding=1).to(dev)
        wide.weight.zero_()
        wide.weight[:, :4].copy_(c0.weight)
        wide.bias.copy_(c0.bias)
        head_pad.stem[0] = wide
        head_pad = head_pad.to(memory_format=torch.channels_last)

    def padded():
        with torch.no_grad(), amp:
            x = torch.cat([image_f32, logits_f16.float(),
                           image_f32.new_zeros(1, 4, S, S)], dim=1)
            x = x.contiguous(memory_format=torch.channels_last)
            return logits_f16 + head_pad.out(head_pad.stem(x))
    report("channels_last + 4->8 padded in", padded)

    # 5. let inductor fuse the norm/activation chain into the convolutions
    try:
        compiled = torch.compile(head_cl, dynamic=False)

        def comp():
            with torch.no_grad(), amp:
                return compiled(log_pre, img_pre)
        report("torch.compile + channels_last", comp)
    except Exception as exc:  # noqa: BLE001
        print(f"  {'torch.compile':34s} failed: {type(exc).__name__}")

    # 6. GroupNorm(1, C) is a whole-tensor reduction, and NHWC makes it worse:
    # with one group the reduction spans every channel, which is contiguous in
    # NCHW and strided in NHWC. But mu and sigma are per-SAMPLE scalars, so once
    # they exist the rest is a per-channel affine -- one reduction pass and one
    # elementwise pass, whatever the layout. Algebraically identical to
    # F.group_norm; measured at 1.8e-07 relative on CPU, i.e. fp32 rounding.
    gn = head_cl.stem[1]

    def gn_fused(x):
        var, mean = torch.var_mean(x, dim=(1, 2, 3), keepdim=True, unbiased=False)
        scale = gn.weight.view(1, -1, 1, 1) * torch.rsqrt(var + gn.eps)
        return torch.addcmul(gn.bias.view(1, -1, 1, 1) - mean * scale, x, scale)

    print("\n--- GroupNorm(1, C) alone " + "-" * 41)
    with torch.no_grad(), amp:
        x_nchw = torch.randn(1, a.channels, S, S, device=dev)
        x_nhwc = x_nchw.contiguous(memory_format=torch.channels_last)
        x_half = x_nhwc.half()
        gn_ref = gn(x_nhwc).float()
        for label, fn, probe in (
                ("F.group_norm NCHW", lambda: gn(x_nchw), x_nchw),
                ("F.group_norm NHWC", lambda: gn(x_nhwc), x_nhwc),
                ("F.group_norm NHWC fp16", lambda: gn(x_half), x_half),
                ("var_mean + addcmul NHWC", lambda: gn_fused(x_nhwc), x_nhwc),
                ("var_mean + addcmul fp16", lambda: gn_fused(x_half), x_half)):
            ms = timed(fn, a.frames)
            d = (fn().float() - gn_ref).abs().max().item()
            print(f"  {label:28s} {ms:7.3f} ms   max|diff| {d:9.3e}")

    # 7. everything that measured well, together
    def best():
        with torch.no_grad(), amp:
            x = torch.cat([image_f32, logits_f16], dim=1).contiguous(
                memory_format=torch.channels_last)
            x = head_cl.stem[0](x)
            x = gn_fused(x)
            x = head_cl.stem[2](x)
            x = head_cl.stem[3](x)
            x = head_cl.stem[4](x)
            return logits_f16 + head_cl.out(x)

    print()
    print(f"  {'variant':34s} {'ms':>8} {'vs base':>9}  {'max|diff|':>10}")
    report("channels_last + fused GroupNorm", best)

    # -- layer by layer again, in the best layout ----------------------------
    print("\n--- inside the head, channels_last " + "-" * 33)
    with torch.no_grad(), amp:
        cur = torch.cat([image_f32, logits_f16], dim=1).contiguous(
            memory_format=torch.channels_last)
        for i, layer in enumerate(head_cl.stem):
            ms = timed((lambda m, t: (lambda: m(t)))(layer, cur), a.frames)
            print(f"  stem[{i}] {type(layer).__name__:22s} {ms:7.3f} ms")
            cur = layer(cur)

    print("\nA variant is only worth taking if its disagreement is small next to")
    print("the residual the head is adding in the first place. Print that scale")
    print("with: (ref - logits).abs().mean()")
    with torch.no_grad():
        residual = (ref - logits_f16.float()).abs().mean().item()
    print(f"mean |residual| this head applies: {residual:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
