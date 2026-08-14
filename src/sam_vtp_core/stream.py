"""Camera-agnostic streaming API.

Deliberately implements no camera I/O. Frames arrive from whatever source you
have (a ROS callback, a RealSense pipeline, a video decoder) as numpy arrays or
torch tensors, and results come back per frame.
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

INPUT_FITS = ("stretch", "crop", "pad")


def _upload(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move to the device, then widen -- not the other way round.

    `t.to(device, torch.float32)` on a uint8 frame casts on the CPU first and
    then moves four bytes per pixel instead of one. Uploading narrow and casting
    on the GPU is bit-identical and moves a quarter of the bytes, which is worth
    having on a board where the CPU is weak and the "PCIe" bus is really the
    same DRAM the GPU is reading from.
    """
    out = t.to(device, non_blocking=True)
    return out if out.dtype == torch.float32 else out.float()


@dataclass
class StreamStepResult:
    frame_index: int
    logits: Union[torch.Tensor, np.ndarray]
    probabilities: Union[torch.Tensor, np.ndarray]
    binary_mask: Union[torch.Tensor, np.ndarray]
    output_hw: Tuple[int, int]
    processing_hw: Tuple[int, int]
    elapsed_ms: float
    graph_status: str
    memory_fill_level: float = 1.0
    fit: Optional[Tuple] = None


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
        input_fit: str = "stretch",
        reset_on_size_change: bool = True,
        return_numpy: bool = False,
        memory_write_every: Optional[int] = None,
        memory_write_interval_s: Optional[float] = None,
        time_model_only: bool = True,
    ):
        """
        Args:
            resize_hw: resolution the model runs at. Fix it for a deployment.
            input_fit: how a frame whose aspect ratio differs from `resize_hw`
                is made to match. See `_plan_fit`; the intrinsics are adjusted
                to agree with whichever you pick, which is the part that is
                easy to get wrong by hand.
            return_numpy: return maps as numpy `(H, W)` arrays instead of torch
                tensors, so a consumer needs no torch layout knowledge.
            time_model_only: synchronise before reading the clock, so
                `elapsed_ms` is the model and not the model plus whatever the
                caller does next. That is what you want when benchmarking. In a
                deployment it drains the device once for the measurement and
                again on the first read of the result, so a consumer that only
                wants the mask can set this False and let the postprocessing
                pipeline behind the forward pass; `elapsed_ms` then reports
                launch time rather than compute time.
            memory_write_every: write memory only every k-th frame.
            memory_write_interval_s: write memory at most this often in
                wall-clock seconds. Takes precedence over `memory_write_every`,
                and suits a camera whose rate wanders; on a camera that holds
                its rate, `memory_write_every` gets the same spacing
                deterministically, which is easier to reason about.

        Why cadence matters: the bank holds a fixed number of *frames*, so the
        time it spans is set by the frame rate. Training used about 0.2 s
        between memory frames, giving the 7-slot bank roughly a 1.2 s window.
        Running at 35 FPS with a write per frame collapses that to about 0.2 s
        and the bank fills with near-duplicate views. Predictions are still
        produced every frame; only the write is gated.

        Gating no longer costs graph replay. The decision is made here, in
        Python, and each of the two outcomes has its own captured graph, so a
        cadence and CUDA graphs coexist -- see `fullgraph.py`.
        """
        if input_fit not in INPUT_FITS:
            raise ValueError(f"input_fit must be one of {INPUT_FITS}, got {input_fit!r}")

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
        self.input_fit = input_fit
        self.reset_on_size_change = bool(reset_on_size_change)
        self.return_numpy = bool(return_numpy)
        self.time_model_only = bool(time_model_only)

        # The camera the caller configured, at its native resolution. The fit
        # derives from this; the model's own `camera` is overwritten to match
        # whatever the fit produces.
        self._native_camera = model.camera
        self._fit: Optional[Tuple] = None

        self._use_cuda_amp = (self.device.type == "cuda") and (self.amp_dtype == "fp16")
        self._frame_index = 0
        self._stream_hw: Optional[Tuple[int, int]] = None

        self.memory_write_interval_s = (None if memory_write_interval_s is None
                                        else float(memory_write_interval_s))
        self._last_memory_write_t: Optional[float] = None
        if memory_write_every is not None:
            self.model.memory_write_every = max(1, int(memory_write_every))

        self._build_runner()
        self.model.reset_memory()

    def _build_runner(self) -> None:
        # Gated writes need a second captured graph for the frames that skip the
        # write. Without one the runner would have to choose between writing
        # anyway and falling back to eager, and both are wrong.
        gated = (self.memory_write_interval_s is not None
                 or int(getattr(self.model, "memory_write_every", 1)) > 1)
        self._runner = SAMVTPFullGraphReplayRunner(
            model=self.model, device=self.device, enabled=self.use_fullgraph,
            capture_warmup=self.capture_warmup,
            min_history_frames=self.fullgraph_min_history,
            capture_nowrite=gated)

    @property
    def graph_status(self) -> str:
        return self._runner.status_note()

    @property
    def memory_fill_level(self) -> float:
        """Fraction of the memory bank holding real frames. Gate planning on this."""
        return self.model.memory_fill_level

    def reset(self) -> None:
        """Drop temporal state. Call on a scene cut or a stream restart.

        This also rebuilds the graph runner, so the next ~17 frames run eagerly
        while the bank refills and the graph is recaptured. Do not wire this to
        a trigger-happy staleness check.
        """
        self.model.reset_memory()
        self._frame_index = 0
        self._stream_hw = None
        self._last_memory_write_t = None
        self._build_runner()

    # -- input fitting -------------------------------------------------------
    def _plan_fit(self, in_h: int, in_w: int):
        """Decide how to reconcile the frame's aspect ratio with `resize_hw`.

        Training used a square 90x90 degree image, so `fx/W == fy/H == 0.5`:
        every pixel subtends the same angle in both axes. A 16:9 camera with
        `fx != fy` does not, and the encoder has never seen anything else.

        Crop and pad therefore target the aspect ratio `fx:fy`, which is the one
        that makes `fx/W == fy/H` after the fit. The final resize to `resize_hw`
        is then anisotropic on purpose: that is exactly what equalises the two
        normalised focal lengths.

          "stretch"  resize straight to the target. Keeps the whole sensor and
                     invents nothing, but angular sampling stays anisotropic:
                     on a D455 the vertical axis ends up ~1.6x magnified
                     relative to training.
          "crop"     centre-crop to fx:fy. Isotropic like training and invents
                     nothing, but throws away field of view, and on a wide
                     camera that is most of the horizontal view.
          "pad"      pad out to fx:fy. Isotropic *and* recovers the training
                     field of view exactly, at the cost of showing the encoder
                     fabricated border content.

        There is no free option, and which wins is empirical on your camera.
        The reason to do it here is that the intrinsics then follow
        automatically: cropping moves the principal point one way and padding
        moves it the other, and getting that wrong tilts every slope estimate.

        Returns (fit, camera) where `fit` is None for a plain resize.
        """
        cam = self._native_camera
        if (cam.width, cam.height) != (in_w, in_h):
            print(f"[sam-vtp] warning: intrinsics are for {cam.width}x{cam.height} "
                  f"but frames are {in_w}x{in_h}. Supply intrinsics at the "
                  f"resolution they were calibrated at.")

        # W/H = fx/fy is equivalent to fx/W = fy/H.
        ar_t, ar_s = cam.fx / cam.fy, in_w / in_h
        if self.input_fit == "stretch" or abs(ar_t - ar_s) < 1e-6:
            return None, cam

        if self.input_fit == "crop":
            if ar_s > ar_t:                       # too wide: trim the sides
                nw = int(round(in_h * ar_t)); x0 = (in_w - nw) // 2
                return ("crop", x0, 0, nw, in_h), cam.crop(x0, 0, nw, in_h)
            nh = int(round(in_w / ar_t)); y0 = (in_h - nh) // 2
            return ("crop", 0, y0, in_w, nh), cam.crop(0, y0, in_w, nh)

        if ar_s > ar_t:                           # too wide: add rows
            nh = int(round(in_w / ar_t)); top = (nh - in_h) // 2
            return ("pad", 0, top, in_w, nh), cam.pad(0, top, in_w, nh)
        nw = int(round(in_h * ar_t)); left = (nw - in_w) // 2
        return ("pad", left, 0, nw, in_h), cam.pad(left, 0, nw, in_h)

    def _apply_fit(self, x: torch.Tensor, pad_mode: str) -> torch.Tensor:
        """Apply the planned crop or pad.

        `pad_mode` is "replicate" for RGB and "zeros" for depth and validity:
        replicating depth into the border would invent surfaces, whereas zero
        depth already means "no measurement".
        """
        if self._fit is None:
            return x
        kind, a, b, w, h = self._fit
        if kind == "crop":
            return x[..., b : b + h, a : a + w]
        left, top = a, b
        pad = (left, w - x.shape[-1] - left, top, h - x.shape[-2] - top)
        if pad_mode == "replicate":
            return F.pad(x, pad, mode="replicate")
        return F.pad(x, pad, mode="constant", value=0.0)

    # -- input marshalling ---------------------------------------------------
    @staticmethod
    def _to_bchw(frame: FrameInput, channels: int = 3,
                 device: Optional[torch.device] = None) -> torch.Tensor:
        """Normalise any accepted layout to [1, channels, H, W].

        `device` moves the frame across *before* the layout work. A camera hands
        over HWC, so reaching BCHW means a transpose; doing it on the CPU costs a
        full copy of the frame in the node's own thread, and on the GPU it costs
        almost nothing. The upload itself is unchanged either way -- same bytes,
        same dtype, just reordered on the other side.
        """
        if isinstance(frame, np.ndarray):
            tensor = torch.from_numpy(np.ascontiguousarray(frame))
        elif torch.is_tensor(frame):
            tensor = frame
        else:
            raise TypeError("frame must be a numpy.ndarray or torch.Tensor")

        if device is not None:
            tensor = tensor.to(device, non_blocking=True)

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
        def arr(x):
            return x if isinstance(x, np.ndarray) else x[0, 0].detach().float().cpu().numpy()
        return {"probability": arr(result.probabilities), "binary": arr(result.binary_mask)}

    # -- stepping ------------------------------------------------------------
    def step(self, frame: FrameInput,
             depth: Optional[FrameInput] = None,
             pose: Optional[FrameInput] = None,
             depth_valid: Optional[FrameInput] = None) -> StreamStepResult:
        """Process one frame.

        Args:
            frame: RGB, in the range the model's `input_normalization_mode` names.
            depth: metric planar z-depth in METRES, resized to match `frame` if
                it does not already.
            pose: `[x y z qx qy qz qw]` scalar-last. Build it with
                `pose_from_optical` from a TF lookup **at the image timestamp**,
                not at processing time: at 0.5 m/s, 60 ms of skew is 3 cm, which
                shows up as smearing in the warped memory.
            depth_valid: optional mask; see `SAMVTP.forward_frame`.
        """
        x = _upload(self._to_bchw(frame, device=self.device), self.device)
        in_h, in_w = int(x.shape[-2]), int(x.shape[-1])

        if self._stream_hw is None:
            self._stream_hw = (in_h, in_w)
            self._fit, cam = self._plan_fit(in_h, in_w)
            self.model.camera = cam
            fh, fv = cam.fov_deg()
            print(f"[sam-vtp] input_fit={self.input_fit}: {in_w}x{in_h} -> "
                  f"{cam.width}x{cam.height} ({fh:.0f}x{fv:.0f} deg FOV), "
                  f"then {self.resize_hw or (in_h, in_w)}. Training was 90x90 deg.")
        elif self._stream_hw != (in_h, in_w):
            if not self.reset_on_size_change:
                raise RuntimeError(
                    f"frame size changed {self._stream_hw} -> {(in_h, in_w)}; "
                    "fix resize_hw or enable reset_on_size_change")
            self.reset()
            self._stream_hw = (in_h, in_w)
            self._fit, cam = self._plan_fit(in_h, in_w)
            self.model.camera = cam

        d = dv = None
        if depth is not None:
            d = _upload(self._to_bchw(depth, channels=1, device=self.device), self.device)
            if d.shape[-2:] != (in_h, in_w):
                d = F.interpolate(d, size=(in_h, in_w), mode="nearest-exact")
        if depth_valid is not None:
            dv = _upload(self._to_bchw(depth_valid, channels=1, device=self.device),
                         self.device)
            if dv.shape[-2:] != (in_h, in_w):
                dv = F.interpolate(dv, size=(in_h, in_w), mode="nearest-exact")

        x = self._apply_fit(x, "replicate")
        if d is not None:
            d = self._apply_fit(d, "zeros")
        if dv is not None:
            dv = self._apply_fit(dv, "zeros")

        if self.resize_hw is not None and x.shape[-2:] != tuple(self.resize_hw):
            th, tw = int(self.resize_hw[0]), int(self.resize_hw[1])
            x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
            if d is not None:
                d = F.interpolate(d, size=(th, tw), mode="nearest-exact")
            if dv is not None:
                dv = F.interpolate(dv, size=(th, tw), mode="nearest-exact")

        out_h, out_w = int(x.shape[-2]), int(x.shape[-1])
        p = None
        if pose is not None:
            p = (torch.as_tensor(pose, dtype=torch.float32)
                 .reshape(1, 7).to(self.device, non_blocking=True))

        proc_h, proc_w = compatible_input_hw(self.model, out_h, out_w)
        if (proc_h, proc_w) != (out_h, out_w):
            pad = (0, proc_w - out_w, 0, proc_h - out_h)
            x = F.pad(x, pad, mode="replicate")
            if d is not None:
                d = F.pad(d, pad, mode="constant", value=0.0)
            if dv is not None:
                dv = F.pad(dv, pad, mode="constant", value=0.0)

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
                                   depth=d, pose=p, depth_valid=dv)

        if self.device.type == "cuda" and self.time_model_only:
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        logits = out["logits"]
        if logits.shape[-2:] != (out_h, out_w):
            logits = logits[..., :out_h, :out_w]
        probs = torch.sigmoid(logits)
        binary = (probs >= self.threshold).to(probs.dtype)

        if self.return_numpy:
            to_np = lambda t: t[0, 0].detach().float().cpu().numpy()  # noqa: E731
            logits, probs, binary = to_np(logits), to_np(probs), to_np(binary)

        result = StreamStepResult(
            frame_index=self._frame_index, logits=logits, probabilities=probs,
            binary_mask=binary, output_hw=(out_h, out_w),
            processing_hw=(proc_h, proc_w), elapsed_ms=elapsed_ms,
            graph_status=self._runner.status_note(),
            memory_fill_level=self.model.memory_fill_level, fit=self._fit)
        self._frame_index += 1
        return result

    def iter_results(self, frames: Iterable[object]) -> Iterator[StreamStepResult]:
        for item in frames:
            yield self.step(*_unpack(item))

    def run_loop(self, frame_provider: FrameProvider,
                 on_result: Optional[StreamCallback] = None,
                 max_frames: Optional[int] = None) -> int:
        """Pull frames until the provider returns None.

        The provider may return a bare RGB frame, or a `(frame, depth, pose)` /
        `(frame, depth, pose, depth_valid)` tuple.
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
    """Accept a bare frame, or a (frame, depth, pose[, depth_valid]) tuple."""
    if isinstance(item, (tuple, list)) and not torch.is_tensor(item):
        return tuple(item)
    return (item,)


def run_stream_loop(processor: SAMVTPStreamProcessor, frame_provider: FrameProvider,
                    on_result: Optional[StreamCallback] = None,
                    max_frames: Optional[int] = None) -> int:
    """Convenience wrapper around `SAMVTPStreamProcessor.run_loop`."""
    return processor.run_loop(frame_provider=frame_provider, on_result=on_result,
                              max_frames=max_frames)
