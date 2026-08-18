"""Building a model from a checkpoint.

The guiding rule: the checkpoint is the authority on what the model is. Flags
passed here are defaults for a fresh model, not overrides for a trained one.
Architecture is read back from the weights themselves (`config_from_state_dict`)
and the load is strict, so a checkpoint that does not match this code raises
instead of quietly producing a differently-shaped model with the same name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .architecture import CameraIntrinsics, SAMVTP, SAMVTPConfig


def resolve_path(path_like: str) -> Path:
    return Path(path_like).expanduser().resolve()


def tune_for_inference(model: Optional[torch.nn.Module] = None, *,
                       channels_last: bool = True,
                       cudnn_benchmark: bool = False) -> str:
    """Apply the deployment-only speedups that do not change the architecture.

    Explicit rather than an import side effect, and explicit rather than folded
    into `build_model`, because both of these move results slightly: a different
    convolution algorithm rounds differently. Do not turn them on for a run whose
    numbers have to reproduce an earlier one exactly.

    channels_last: NHWC for the whole model. Narrow convolutions are badly
        served by NCHW -- on an Orin Nano the refinement head's two 3x3
        convolutions took 9.9 ms in NCHW and 1.6 ms in NHWC -- and the memory
        encoder and the decoder's transposed convolutions pay the same tax.
        Measured 50.3 -> 47.4 ms per frame on the whole model. Attention is
        indifferent to it. On by default because it measured as a clear win with
        no downside beyond that rounding.

    cudnn_benchmark: time the available convolution algorithms on first use and
        keep the fastest. Wants stable shapes, which hold here, and `warmup`
        pays for the search off the clock. OFF by default: it measured as
        exactly neutral on the refinement head (14.084 vs 14.081 ms,
        bit-identical output), so it is unproven on this model and an unproven
        source of run-to-run variation is not worth taking for free.

    Returns a one-line description, worth logging next to the model banner.
    """
    if not torch.cuda.is_available():
        return "tune_for_inference: no CUDA, nothing to set"
    notes = []
    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        notes.append("cudnn.benchmark=True")
    if channels_last and model is not None:
        model.to(memory_format=torch.channels_last)
        notes.append("channels_last")
    elif channels_last:
        notes.append("channels_last SKIPPED (no model passed)")
    return f"tune_for_inference: {', '.join(notes) or 'nothing enabled'} (not bit-exact)"


def load_checkpoint_state(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Just the weights, for callers that do not care about training metadata."""
    return read_checkpoint(path, map_location=map_location)[0]


def read_checkpoint(path: str, map_location: str = "cpu"
                    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Return `(state_dict, training_config)` from a SAM-VTP checkpoint.

    The training config is whatever the trainer embedded. It is advisory --
    every architectural fact is re-derived from the tensors -- except for
    `multimask_output`, which leaves no trace in the weights and so can only
    come from here.
    """
    checkpoint_path = resolve_path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    try:
        raw = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 1.13
        raw = torch.load(checkpoint_path, map_location=map_location)

    if not isinstance(raw, dict):
        raise TypeError(f"expected a dict checkpoint, got {type(raw).__name__}")
    state = raw.get("model_state_dict", raw.get("model", raw))
    train_cfg = raw.get("config", {}) if isinstance(raw.get("config", {}), dict) else {}
    return state, train_cfg


def build_model(
    sam_vtp_checkpoint_path: str,
    device: torch.device = torch.device("cpu"),
    *,
    normalization_mode: str = "0_255",
    memory_write_every: int = 1,
    camera: Optional[CameraIntrinsics] = None,
    multimask: Optional[bool] = None,
    components: Optional[Dict[str, bool]] = None,
    **config_overrides: Any,
) -> SAMVTP:
    """Build SAM-VTP and load `sam_vtp_checkpoint_path` into it.

    Args:
        normalization_mode: the range your images arrive in -- `"0_255"`,
            `"0_1"`, `"-1_1"`, or `"auto"`. Pin it explicitly in deployment;
            `"auto"` infers from each batch's range and a dark frame can fool it.
        memory_write_every: write memory only every k-th frame. The bank holds a
            fixed number of frames, so the time it spans is set by the frame
            rate; k>1 restores a training-like window on a fast camera. Every
            frame still gets a prediction.
        camera: intrinsics for the depth used by the geometry path. Defaults to
            the TartanGround camera the released checkpoints were trained on --
            which is almost certainly not yours. See CameraIntrinsics.
        multimask: normally read from the checkpoint's training config. It is
            invisible in the weights, so a checkpoint without a config and
            without this flag falls back to False.
        components: runtime switches, e.g. `{"memory": False}` for the
            frame-wise ablation or `{"geometry": False}` for RGB-only. Keys are
            `memory`, `geometry`, `refine`. These select which of the
            checkpoint's weights are USED; they cannot add a component the
            checkpoint lacks. See `SAMVTP.set_components`.
        **config_overrides: forwarded to SAMVTPConfig. Overriding an
            architectural field here will make the checkpoint fail to load.
    """
    state, train_cfg = read_checkpoint(sam_vtp_checkpoint_path, map_location="cpu")

    if multimask is None:
        multimask = bool(train_cfg.get("multimask_output", False))
    cfg = SAMVTP.config_from_state_dict(state, multimask=bool(multimask), **config_overrides)

    model = SAMVTP(cfg)
    model.load_checkpoint(state)
    model.input_normalization_mode = normalization_mode
    model.memory_write_every = max(1, int(memory_write_every))
    if camera is not None:
        model.camera = camera
    if components:
        model.set_components(**components)
    return model.to(device).eval()


def describe(model: SAMVTP) -> str:
    """One-line summary, for logs that need to record what actually ran.

    Ends with an explicit ABLATED clause whenever a component is switched away
    from how the weights were trained. Anything that logs this line can then
    show, rather than assume, that a number came from the full model.
    """
    n = sum(p.numel() for p in model.parameters())
    active = model.active_components
    bits = [f"SAM-VTP {n / 1e6:.1f}M params",
            f"memory {'on, ' + str(model.num_maskmem) + ' frames' if active['memory'] else 'OFF (frame-wise)'}",
            f"geometry {'on' if active['geometry'] else 'off'}",
            f"refine head {'on' if active['refine'] else 'off'}",
            f"multimask {'on' if model.multimask_output else 'off'}"]
    if model.memory_write_every > 1:
        bits.append(f"memory write every {model.memory_write_every}")
    line = ", ".join(bits)

    deviations = model.component_deviations
    if deviations:
        trained = model.trained_components
        detail = ", ".join(
            f"{k} {'on' if v else 'off'} (trained {'on' if trained[k] else 'off'})"
            for k, v in sorted(deviations.items()))
        line += f"  [ABLATED: {detail}]"
    return line
