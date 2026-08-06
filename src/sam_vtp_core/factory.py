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
    return model.to(device).eval()


def describe(model: SAMVTP) -> str:
    """One-line summary, for logs that need to record what actually ran."""
    n = sum(p.numel() for p in model.parameters())
    bits = [f"SAM-VTP {n / 1e6:.1f}M params",
            f"memory {model.num_maskmem} frames",
            f"warp {'on' if model.warp_memory else 'off'}",
            f"geometry {'on' if model.use_geometry else 'off'}",
            f"refine head {'on' if model.refine_head is not None else 'off'}",
            f"multimask {'on' if model.multimask_output else 'off'}"]
    if model.memory_write_every > 1:
        bits.append(f"memory write every {model.memory_write_every}")
    return ", ".join(bits)
