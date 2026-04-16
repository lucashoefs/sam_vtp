from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import time
from typing import Callable, Dict, Iterable, Iterator, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .fullgraph import SAMVTPFullGraphReplayRunner
from .inference import _sam2_compatible_input_hw
from .model import SAMVTP


FrameInput = Union[np.ndarray, torch.Tensor]
FrameProvider = Callable[[], Optional[FrameInput]]
StreamCallback = Callable[["StreamStepResult"], None]


@dataclass
class StreamStepResult:
    """Single step output for streaming traversability inference."""

    frame_index: int
    logits: torch.Tensor
    probabilities: torch.Tensor
    binary_mask: torch.Tensor
    output_hw: Tuple[int, int]
    processing_hw: Tuple[int, int]
    elapsed_ms: float
    graph_status: str


class SAMVTPStreamProcessor:
    """Camera-agnostic real-time loop infrastructure for SAM-VTP inference.

    This class intentionally does not implement camera I/O. Instead, it provides
    a generic stream API that accepts frames from any source as numpy arrays or
    torch tensors and produces traversability maps continuously.
    """

    def __init__(
        self,
        model: SAMVTP,
        use_rolling_memory: bool = True,
        use_fullgraph: Optional[bool] = None,
        fullgraph_min_history: int = 16,
        capture_warmup: int = 0,
        amp_dtype: str = "fp16",
        threshold: float = 0.5,
        resize_hw: Optional[Tuple[int, int]] = None,
        reset_on_size_change: bool = True,
    ):
        self.model = model
        self.device = next(model.parameters()).device

        if use_fullgraph is None:
            use_fullgraph = self.device.type == "cuda"

        self.use_rolling_memory = bool(use_rolling_memory)
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

        self._build_runner()
        self.model.reset_memory()

    def _build_runner(self) -> None:
        self._runner = SAMVTPFullGraphReplayRunner(
            model=self.model,
            device=self.device,
            enabled=self.use_fullgraph,
            capture_warmup=self.capture_warmup,
            min_history_frames=self.fullgraph_min_history,
        )

    @property
    def graph_status(self) -> str:
        return self._runner.status_note()

    def reset(self) -> None:
        self.model.reset_memory()
        self._frame_index = 0
        self._stream_hw = None
        self._build_runner()

    @staticmethod
    def _to_bchw(frame: FrameInput) -> torch.Tensor:
        if isinstance(frame, np.ndarray):
            tensor = torch.from_numpy(frame)
        elif torch.is_tensor(frame):
            tensor = frame
        else:
            raise TypeError("frame must be a numpy.ndarray or torch.Tensor")

        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError("batched input must have batch size 1 for streaming")
            if tensor.shape[1] == 3:
                bchw = tensor
            elif tensor.shape[-1] == 3:
                bchw = tensor.permute(0, 3, 1, 2)
            else:
                raise ValueError("4D frame must be [1, 3, H, W] or [1, H, W, 3]")
        elif tensor.ndim == 3:
            if tensor.shape[0] == 3:
                bchw = tensor.unsqueeze(0)
            elif tensor.shape[-1] == 3:
                bchw = tensor.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError("3D frame must be [3, H, W] or [H, W, 3]")
        else:
            raise ValueError("frame must be 3D or 4D")

        return bchw.contiguous()

    @staticmethod
    def to_numpy_maps(result: StreamStepResult) -> Dict[str, np.ndarray]:
        return {
            "probability": result.probabilities[0, 0].detach().float().cpu().numpy(),
            "binary": result.binary_mask[0, 0].detach().float().cpu().numpy(),
        }

    def _prepare_frame(self, frame: FrameInput) -> torch.Tensor:
        x = self._to_bchw(frame).to(device=self.device, dtype=torch.float32, non_blocking=True)
        if self.resize_hw is not None:
            target_h, target_w = int(self.resize_hw[0]), int(self.resize_hw[1])
            if x.shape[-2:] != (target_h, target_w):
                x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return x

    def step(self, frame: FrameInput) -> StreamStepResult:
        x = self._prepare_frame(frame)

        out_h, out_w = int(x.shape[-2]), int(x.shape[-1])
        current_hw = (out_h, out_w)

        if self._stream_hw is None:
            self._stream_hw = current_hw
        elif self._stream_hw != current_hw:
            if not self.reset_on_size_change:
                raise RuntimeError(
                    "Frame size changed from {} to {}. Provide a fixed resize_hw or enable reset_on_size_change.".format(
                        self._stream_hw,
                        current_hw,
                    )
                )
            self.reset()
            self._stream_hw = current_hw

        proc_h, proc_w = _sam2_compatible_input_hw(self.model, out_h, out_w)
        if (proc_h != out_h) or (proc_w != out_w):
            x = F.pad(x, (0, proc_w - out_w, 0, proc_h - out_h), mode="replicate")

        is_init = self._frame_index == 0

        t0 = time.perf_counter()
        amp_context = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if self._use_cuda_amp
            else nullcontext()
        )
        with torch.no_grad():
            with amp_context:
                out = self._runner.run(
                    x,
                    is_init_frame=is_init,
                    use_rolling_memory=self.use_rolling_memory,
                )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        logits = out["logits"]
        if logits.shape[-2:] != (out_h, out_w):
            logits = logits[..., :out_h, :out_w]

        probs = torch.sigmoid(logits)
        binary = (probs >= self.threshold).to(dtype=probs.dtype)

        result = StreamStepResult(
            frame_index=self._frame_index,
            logits=logits,
            probabilities=probs,
            binary_mask=binary,
            output_hw=(out_h, out_w),
            processing_hw=(proc_h, proc_w),
            elapsed_ms=elapsed_ms,
            graph_status=self._runner.status_note(),
        )

        self._frame_index += 1
        return result

    def iter_results(self, frames: Iterable[FrameInput]) -> Iterator[StreamStepResult]:
        for frame in frames:
            yield self.step(frame)

    def run_loop(
        self,
        frame_provider: FrameProvider,
        on_result: Optional[StreamCallback] = None,
        max_frames: Optional[int] = None,
    ) -> int:
        processed = 0
        while True:
            frame = frame_provider()
            if frame is None:
                break

            result = self.step(frame)
            if on_result is not None:
                on_result(result)

            processed += 1
            if (max_frames is not None) and (processed >= int(max_frames)):
                break

        return processed


def run_stream_loop(
    processor: SAMVTPStreamProcessor,
    frame_provider: FrameProvider,
    on_result: Optional[StreamCallback] = None,
    max_frames: Optional[int] = None,
) -> int:
    """Convenience wrapper around SAMVTPStreamProcessor.run_loop."""
    return processor.run_loop(frame_provider=frame_provider, on_result=on_result, max_frames=max_frames)
