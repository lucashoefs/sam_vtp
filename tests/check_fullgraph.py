"""Verify the CUDA-graph runner against the eager path.

fullgraph.py is a second implementation of forward_frame, carrying its own
copies of divergences 1, 2, 3 and 5. This checks the two agree.

The two paths are compared against each other rather than against stored data,
so this needs nothing but a checkpoint and a GPU. The clip has to outrun
min_history_frames (16), below which the runner falls back to eager and the test
would only measure the path it already trusts.

A graph replay is not required to be bit-identical to the same ops run eagerly:
capture can fix a different kernel choice or accumulation order. Ulp-level
differences on replayed frames, with warmup frames exact, are the expected
outcome rather than a bug. A structural error looks different -- it appears at
the first replayed frame and is orders of magnitude larger.

Usage:
    python tests/check_fullgraph.py --checkpoint sam_vtp.pt
    python tests/check_fullgraph.py --checkpoint sam_vtp.pt --variant no_memory
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam_vtp import build_model  # noqa: E402
from sam_vtp.fullgraph import SAMVTPFullGraphReplayRunner  # noqa: E402


# ---------------------------------------------------------------------------
# Determinism and input
# ---------------------------------------------------------------------------
def pin_determinism(device: torch.device, threads: int) -> None:
    """Remove every source of run-to-run variation we can reach.

    cudnn.benchmark is the important one: left on, cuDNN autotunes convolution
    algorithms on first use and can pick differently between processes, which
    makes --selftest fail for reasons unrelated to the model. deterministic=True
    additionally rules out the non-deterministic algorithm choices cuDNN would
    otherwise be free to make.
    """
    if threads > 0:
        torch.set_num_threads(threads)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # TF32 trades mantissa bits for speed on matmul and convolution. It is
        # deterministic, but it is a different number than fp32, so pin it
        # explicitly rather than inheriting whatever the torch default is this
        # release -- that default has changed before.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() "
                         "is False. Check you installed a CUDA build.")
    return torch.device(name)


def make_sequence(frames: int, hw: int, seed: int = 20260907,
                  device: torch.device | str = "cpu"):
    """A deterministic clip that moves.

    Built on CPU with a CPU generator regardless of target device, then moved.
    A CUDA generator with the same seed produces different numbers, which would
    make a CPU run and a CUDA run describe different inputs.

    Motion matters: with identical frames a wrong temporal encoding still gives
    identical output, so a static clip would pass a broken memory read. Depth is
    a plausible ground plane in METRES; geometry_channels treats non-positive
    depth as missing.
    """
    g = torch.Generator().manual_seed(seed)
    v = torch.linspace(0.0, 1.0, hw).view(hw, 1)
    u = torch.linspace(0.0, 1.0, hw).view(1, hw)

    images, depths = [], []
    for t in range(frames):
        phase = 0.35 * t
        base = torch.stack([
            0.5 + 0.5 * torch.sin(6.0 * u + phase).expand(hw, hw),
            0.5 + 0.5 * torch.cos(5.0 * v + 0.5 * phase).expand(hw, hw),
            0.5 + 0.5 * torch.sin(3.0 * (u + v) - phase).expand(hw, hw),
        ])
        noise = 0.05 * torch.rand(3, hw, hw, generator=g)
        images.append(((base + noise).clamp(0, 1) * 255.0).unsqueeze(0).to(device))

        d = 2.0 + 18.0 * (1.0 - v).expand(hw, hw)
        d = d + 0.4 * torch.sin(4.0 * u + phase).expand(hw, hw)
        d = d + 0.02 * torch.rand(hw, hw, generator=g)
        depths.append(d.clamp(min=0.3).view(1, 1, hw, hw).to(device))

    return images, depths

SWITCHES = {
    "full": {},
    "no_memory": {"memory": False},
    "no_geometry": {"geometry": False},
    "no_refine": {"refine": False},
}


def run_eager(model, images, depths):
    model.reset_memory()
    out = []
    with torch.no_grad():
        for t, (img, dep) in enumerate(zip(images, depths)):
            r = model.forward_frame(img, is_init_frame=(t == 0),
                                    depth=dep if model.needs_depth else None)
            out.append(r["logits"].detach().float().cpu())
    return out


def run_graph(model, runner, images, depths):
    model.reset_memory()
    out = []
    with torch.no_grad():
        for t, (img, dep) in enumerate(zip(images, depths)):
            r = runner.run(img, is_init_frame=(t == 0),
                           depth=dep if model.needs_depth else None)
            # Replay returns the STATIC output buffer, overwritten next frame.
            # Cloning is mandatory, not hygiene.
            out.append(r["logits"].detach().float().cpu().clone())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--variant", default="full", choices=sorted(SWITCHES))
    ap.add_argument("--frames", type=int, default=40,
                    help="needs to exceed min_history_frames (16) by enough to "
                         "exercise replay and at least one bank eviction")
    ap.add_argument("--hw", type=int, default=512)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--capture-warmup", type=int, default=3)
    args = ap.parse_args()

    device = resolve_device("cuda")
    pin_determinism(device, threads=1)

    images, depths = make_sequence(args.frames, args.hw, device=device)

    def build():
        m = build_model(args.checkpoint, device=device)
        if SWITCHES[args.variant]:
            m.set_components(**SWITCHES[args.variant])
        return m

    eager_model = build()
    print(eager_model.describe())
    eager_out = run_eager(eager_model, images, depths)

    graph_model = build()
    runner = SAMVTPFullGraphReplayRunner(
        graph_model, device=device, enabled=True,
        capture_warmup=args.capture_warmup)
    graph_out = run_graph(graph_model, runner, images, depths)
    print(runner.status_note())

    if not runner._captured:
        print("\nFAIL: capture never happened, so nothing was verified.")
        if runner._capture_error:
            print(f"  capture error: {runner._capture_error}")
        print(f"  min_history_frames={runner.min_history_frames}, "
              f"frames={args.frames}")
        return 1

    warm = runner.min_history_frames
    print(f"\nmin_history_frames={warm} -> frames 0-{warm - 1} eager, "
          f"{warm}-{args.frames - 1} replayed\n")

    worst_warm, worst_replay, first_bad = 0.0, 0.0, None
    for t, (a, b) in enumerate(zip(eager_out, graph_out)):
        d = (a - b).abs().max().item()
        if t < warm:
            worst_warm = max(worst_warm, d)
        else:
            worst_replay = max(worst_replay, d)
            if d > args.tol and first_bad is None:
                first_bad = (t, d)
        if t < 3 or t == warm - 1 or t >= warm and t < warm + 3:
            print(f"  frame {t:3d} {'eager ' if t < warm else 'replay'}  {d:.3e}")

    print(f"\nworst over warmup frames:  {worst_warm:.3e}")
    print(f"worst over replayed frames: {worst_replay:.3e}")

    if worst_warm > 0.0:
        print("\nFAIL: warmup frames differ, and those run eagerly in BOTH "
              "passes.\nThat is not a graph problem -- the two models are not "
              "identical. Check build_model determinism before looking at "
              "fullgraph.py.")
        return 1

    if first_bad is None:
        print(f"\nPASS (tol {args.tol:g})")
        return 0

    t, d = first_bad
    print(f"\nFAIL: first replayed frame over tolerance is {t} at {d:.3e}")
    print("\nreading this:")
    if t == warm:
        print(f"  Frame {warm} is the first replayed frame, so the fault is in "
              f"the handover:\n"
              f"  _sync_tensor_state_from_model slot ordering, the "
              f"maskmem_tpos_enc\n"
              f"  indexing in _memory_conditioned_pix_feat, or step_idx "
              f"initialisation.")
    else:
        print(f"  Frames {warm}-{t - 1} replay correctly and {t} does not, so "
              f"the state\n"
              f"  UPDATE is wrong rather than the state read: the bank shift in\n"
              f"  _forward_tensorized_state, or step_idx not advancing.")
    if d < 1e-3:
        print(f"\n  At {d:.1e} this may be capture-vs-eager kernel choice rather "
              f"than a\n  bug. Raise --tol and check whether the error stays "
              f"flat across frames\n  (rounding) or grows (a real state error "
              f"compounding).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
