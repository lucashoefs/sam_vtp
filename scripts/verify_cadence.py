#!/usr/bin/env python3
"""The graph runner's write gate must agree with the eager one, frame for frame.

CUDA graphs need a GPU, but the *decision* about which graph to replay is plain
Python and can be checked here. The failure this guards against is a cadence
that falls out of phase at the eager-to-replay handover: writes would still
happen at the right rate, just on different frames than the warmup put them on,
and the bank's temporal spacing would be subtly wrong with nothing to show for
it.
"""
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from sam_vtp_core.architecture import SAMVTP, SAMVTPConfig  # noqa: E402
from sam_vtp_core.fullgraph import SAMVTPFullGraphReplayRunner  # noqa: E402

EVERY = 3
N = 24
H = W = 128

cfg = SAMVTPConfig()


def mk(t):
    g = torch.Generator().manual_seed(500 + t)
    img = torch.rand(1, 3, H, W, generator=g) * 255
    dep = torch.rand(1, 1, H, W, generator=g) * 20 + 1.0
    return img, dep


# -- 1. what eager does -------------------------------------------------------
eager = SAMVTP(cfg).eval()
eager.memory_write_every = EVERY
eager_writes = []
with torch.no_grad():
    for t in range(N):
        img, dep = mk(t)
        before = eager.frame_idx
        eager.forward_frame(img, is_init_frame=(t == 0), depth=dep)
        eager_writes.append(eager.frame_idx != before)

print(f"eager writes on   : {[i for i, w in enumerate(eager_writes) if w]}")
print(f"  frame_idx={eager.frame_idx}  _frames_seen={eager._frames_seen}")

# -- 2. what the runner's gate would decide, given the same counter -----------
# The runner is asked the same question in the same order, on a model whose
# counter has been advanced by an identical eager warmup. Replay is not needed:
# only the gate is under test.
probe = SAMVTP(cfg).eval()
probe.memory_write_every = EVERY
runner = SAMVTPFullGraphReplayRunner(probe, torch.device("cpu"), enabled=False,
                                     capture_nowrite=True)
gate_writes = [runner._writes_this_frame(True) for _ in range(N)]
print(f"runner gate writes: {[i for i, w in enumerate(gate_writes) if w]}")

assert gate_writes == eager_writes, (
    f"cadence disagrees:\n  eager  {eager_writes}\n  runner {gate_writes}")
print("\n1. gate matches eager frame for frame        OK")

# -- 3. phase must survive a handover part-way through ------------------------
# Warm up eagerly for `cut` frames, then hand over to the gate. The writes must
# continue on the same frames they would have without the handover.
for cut in (1, 2, 4, 7):
    m = SAMVTP(cfg).eval()
    m.memory_write_every = EVERY
    mixed = []
    with torch.no_grad():
        for t in range(cut):
            img, dep = mk(t)
            before = m.frame_idx
            m.forward_frame(img, is_init_frame=(t == 0), depth=dep)
            mixed.append(m.frame_idx != before)
    r = SAMVTPFullGraphReplayRunner(m, torch.device("cpu"), enabled=False,
                                    capture_nowrite=True)
    mixed += [r._writes_this_frame(True) for _ in range(N - cut)]
    assert mixed == eager_writes, (
        f"handover at {cut} shifted the cadence:\n"
        f"  want {eager_writes}\n  got  {mixed}")
print("2. phase survives handover at frames 1,2,4,7  OK")

# -- 4. the stream processor keeps graphs on when a cadence is set -------------
from sam_vtp_core.stream import SAMVTPStreamProcessor  # noqa: E402

sp = SAMVTPStreamProcessor(SAMVTP(cfg).eval(), use_fullgraph=True,
                           memory_write_every=EVERY, resize_hw=(H, W))
assert sp.use_fullgraph, "a write cadence must no longer disable graph mode"
assert sp._runner._capture_nowrite, "gated writes need the second graph"
print("3. cadence no longer disables graph mode      OK")

sp2 = SAMVTPStreamProcessor(SAMVTP(cfg).eval(), use_fullgraph=True,
                            memory_write_interval_s=0.2, resize_hw=(H, W))
assert sp2.use_fullgraph and sp2._runner._capture_nowrite
print("4. wall-clock gating also keeps graph mode    OK")

sp3 = SAMVTPStreamProcessor(SAMVTP(cfg).eval(), use_fullgraph=True, resize_hw=(H, W))
assert not sp3._runner._capture_nowrite, (
    "an ungated runner should not pay for a second graph")
print("5. ungated runner captures one graph only     OK")

# -- 5. an ungated runner must refuse a non-write frame rather than guess ------
m = SAMVTP(cfg).eval()
r = SAMVTPFullGraphReplayRunner(m, torch.device("cpu"),
                                enabled=False, capture_nowrite=False)
r._captured = True
r._captured_component_epoch = m.component_epoch   # or the epoch check fires first
r._graph_nowrite = None
r.enabled = True
r._state = object()
try:
    r.run(torch.zeros(1, 3, H, W), is_init_frame=False, update_memory=False)
    raise AssertionError("should have refused")
except RuntimeError as exc:
    assert "capture_nowrite" in str(exc), exc
print("6. ungated runner refuses update_memory=False OK")

print("\nall cadence properties hold (CUDA replay itself still needs the GPU:"
      "\n  python scripts/verify_fullgraph_parity.py --memory-write-every 3)")
