"""Camera-agnostic streaming API.

Deliberately implements no camera I/O. Frames arrive from whatever source you
have -- a ROS callback, a RealSense pipeline, a video decoder -- as numpy arrays
or torch tensors, and results come back per frame.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .architecture import SAMVTP
from .fullgraph import SAMVTPFullGraphReplayRunner
from .inference import compatible_input_hw

FrameInput = Union[np.ndarray, torch.Tensor]
FrameProvider = Callable[[], Optional[object]]
StreamCallback = Callable[["StreamStepResult"], None]


@dataclass
class StreamStepResult:
    frame_index: int
    logits: torch.Tensor
    probabilities: torch.Tensor
    binary_mask: torch.Tensor
    output_hw: Tuple[int, int]
    processing_hw: Tuple[int, int]
    elapsed_ms: float
    graph_status: str


class SAMVTPStreamProcessor:
    """Drives SAM-VTP over a live stream, holding the temporal memory."""

    def __init__(
        self,
        model: SAMVTP,
        use_fullgraph: Optional[bool] = None,
        fullgraph_min_history: int = 16,
        capture_warmup: int = 0,
        amp_dtype: str = "fp16",
        threshold: float = 0.5,
        resize_hw: Optional[Tuple[int, int]] = None,
        reset_on_size_change: bool = True,
        memory_write_every: Optional[int] = None,
        memory_write_interval_s: Optional[float] = None,
    ):
        """
        Args:
            memory_write_every: write memory only every k-th frame.
            memory_write_interval_s: write memory at most this often in
                wall-clock seconds. Takes precedence over `memory_write_every`
                and is the better choice for a live camera, whose rate varies.

        Why cadence matters: the bank holds a fixed number of *frames*, so the
        time it spans is set by the frame rate. Training used ~0.2 s between
        memory frames, giving the 7-slot bank a ~1.2 s window. Running at 35 FPS
        with a write per frame collapses that to ~0.2 s and the bank fills with
        near-duplicate views. Predictions are still produced every frame; only
        the write is gated.
        """
        self.model = model
        self.device = next(model.parameters()).device
        if use_fullgraph is None:
            use_fullgraph = self.device.type == "cuda"

        self.use_fullgraph = bool(use_fullgraph)
        self.fullgraph_min_history = int(fullgraph_min_history)
        self.capture_warmup = int(capture_warmup)
        self.amp_dtype = str(amp_dtype)
        self.threshold = float(threshold)
        self.resize_hw = resize_hw
        self.reset_on_size_change = bool(reset_on_size_change)

        self._use_cuda_amp = (self.device.type == "cuda") and (self.amp_dtype == "fp16")
        self._frame_index = 0
        self._stream_hw: Optional[Tuple[int, int]] = None

        self.memory_write_interval_s = (None if memory_write_interval_s is None
                                        else float(memory_write_interval_s))
        self._last_memory_write_t: Optional[float] = None
        if memory_write_every is not None:
            self.model.memory_write_every = max(1, int(memory_write_every))
        if self.memory_write_interval_s is not None:
            # Time-based gating needs per-frame control flow, which a captured
            # graph cannot provide. Make that explicit rather than silently
            # ignoring the interval.
            self.use_fullgraph = False

        self._build_runner()
        self.model.reset_memory()

    def _build_runner(self) -> None:
        self._runner = SAMVTPFullGraphReplayRunner(
            model=self.model, device=self.device, enabled=self.use_fullgraph,
            capture_warmup=self.capture_warmup,
            min_history_frames=self.fullgraph_min_history)

    @property
    def graph_status(self) -> str:
        return self._runner.status_note()

    def reset(self) -> None:
        """Drop temporal state. Call on a scene cut or a stream restart."""
        self.model.reset_memory()
        self._frame_index = 0
        self._stream_hw = None
        self._last_memory_write_t = None
        self._build_runner()

    # -- input marshalling ---------------------------------------------------
    @staticmethod
    def _to_bchw(frame: FrameInput, channels: int = 3) -> torch.Tensor:
        if isinstance(frame, np.ndarray):
            tensor = torch.from_numpy(frame)
        elif torch.is_tensor(frame):
            tensor = frame
        else:
            raise TypeError("frame must be a numpy.ndarray or torch.Tensor")

        if tensor.ndim == 2 and channels == 1:
            return tensor[None, None].contiguous()
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError("streaming expects batch size 1")
            if tensor.shape[1] == channels:
                return tensor.contiguous()
            if tensor.shape[-1] == channels:
                return tensor.permute(0, 3, 1, 2).contiguous()
            raise ValueError(f"4D input must be [1,{channels},H,W] or [1,H,W,{channels}]")
        if tensor.ndim == 3:
            if tensor.shape[0] == channels:
                return tensor.unsqueeze(0).contiguous()
            if tensor.shape[-1] == channels:
                return tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
            raise ValueError(f"3D input must be [{channels},H,W] or [H,W,{channels}]")
        raise ValueError("input must be 2D, 3D or 4D")

    @staticmethod
    def to_numpy_maps(result: StreamStepResult) -> Dict[str, np.ndarray]:
        return {"probability": result.probabilities[0, 0].detach().float().cpu().numpy(),
                "binary": result.binary_mask[0, 0].detach().float().cpu().numpy()}

    # -- stepping ------------------------------------------------------------
    def step(self, frame: FrameInput,
             depth: Optional[FrameInput] = None,
             pose: Optional[FrameInput] = None) -> StreamStepResult:
        """Process one frame.

        Args:
            frame: RGB, in the range the model's `input_normalization_mode` names.
            depth: metric planar z-depth in METRES, same resolution as `frame`
                (it is resized to match if not).
            pose: camera-to-world `[x y z qx qy qz qw]`, scalar-last, NED.

        Required for geometry checkpoints; `forward_frame` raises if missing.
        """
        x = self._to_bchw(frame).to(self.device, torch.float32, non_blocking=True)
        if self.resize_hw is not None and x.shape[-2:] != tuple(self.resize_hw):
            x = F.interpolate(x, size=(int(self.resize_hw[0]), int(self.resize_hw[1])),
                              mode="bilinear", align_corners=False)

        out_h, out_w = int(x.shape[-2]), int(x.shape[-1])
        if self._stream_hw is None:
            self._stream_hw = (out_h, out_w)
        elif self._stream_hw != (out_h, out_w):
            if not self.reset_on_size_change:
                raise RuntimeError(
                    f"frame size changed {self._stream_hw} -> {(out_h, out_w)}; "
                    "fix resize_hw or enable reset_on_size_change")
            self.reset()
            self._stream_hw = (out_h, out_w)

        d = None
        if depth is not None:
            d = self._to_bchw(depth, channels=1).to(self.device, torch.float32, non_blocking=True)
            if d.shape[-2:] != (out_h, out_w):
                d = F.interpolate(d, size=(out_h, out_w), mode="nearest-exact")
        p = None
        if pose is not None:
            p = (torch.as_tensor(pose, dtype=torch.float32)
                 .reshape(1, 7).to(self.device, non_blocking=True))

        proc_h, proc_w = compatible_input_hw(self.model, out_h, out_w)
        if (proc_h, proc_w) != (out_h, out_w):
            pad = (0, proc_w - out_w, 0, proc_h - out_h)
            x = F.pad(x, pad, mode="replicate")
            if d is not None:
                d = F.pad(d, pad, mode="replicate")

        is_init = self._frame_index == 0
        t0 = time.perf_counter()

        update_memory = True
        if self.memory_write_interval_s is not None and not is_init:
            if self._last_memory_write_t is not None:
                update_memory = (t0 - self._last_memory_write_t) >= self.memory_write_interval_s
        if update_memory:
            self._last_memory_write_t = t0

        amp_context = (torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                       if self._use_cuda_amp else nullcontext())
        with torch.no_grad(), amp_context:
            out = self._runner.run(x, is_init_frame=is_init, update_memory=update_memory,
                                   depth=d, pose=p)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        logits = out["logits"]
        if logits.shape[-2:] != (out_h, out_w):
            logits = logits[..., :out_h, :out_w]
        probs = torch.sigmoid(logits)

        result = StreamStepResult(
            frame_index=self._frame_index, logits=logits, probabilities=probs,
            binary_mask=(probs >= self.threshold).to(probs.dtype),
            output_hw=(out_h, out_w), processing_hw=(proc_h, proc_w),
            elapsed_ms=elapsed_ms, graph_status=self._runner.status_note())
        self._frame_index += 1
        return result

    def iter_results(self, frames: Iterable[object]) -> Iterator[StreamStepResult]:
        for item in frames:
            yield self.step(*_unpack(item))

    def run_loop(self, frame_provider: FrameProvider,
                 on_result: Optional[StreamCallback] = None,
                 max_frames: Optional[int] = None) -> int:
        """Pull frames until the provider returns None.

        The provider may return a bare RGB frame, or `(frame, depth, pose)` for
        geometry models.
        """
        processed = 0
        while True:
            item = frame_provider()
            if item is None:
                break
            result = self.step(*_unpack(item))
            if on_result is not None:
                on_result(result)
            processed += 1
            if max_frames is not None and processed >= int(max_frames):
                break
        return processed


def _unpack(item: object):
    """Accept either a bare frame or a (frame, depth, pose) tuple."""
    if isinstance(item, (tuple, list)) and not torch.is_tensor(item):
        return tuple(item)
    return (item,)


def run_stream_loop(processor: SAMVTPStreamProcessor, frame_provider: FrameProvider,
                    on_result: Optional[StreamCallback] = None,
                    max_frames: Optional[int] = None) -> int:
    """Convenience wrapper around `SAMVTPStreamProcessor.run_loop`."""
    return processor.run_loop(frame_provider=frame_provider, on_result=on_result,
                              max_frames=max_frames)
