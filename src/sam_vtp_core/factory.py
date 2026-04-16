from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .model import SAMVTP


def resolve_path(path_like: str) -> Path:
    return Path(path_like).expanduser().resolve()


def load_checkpoint_state(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    checkpoint_path = resolve_path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))

    try:
        raw = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        raw = torch.load(checkpoint_path, map_location=map_location)

    return raw.get("model_state_dict", raw.get("model", raw))


def build_model(
    sam2_cfg_path: str,
    sam_vtp_checkpoint_path: Optional[str],
    device: torch.device,
    num_maskmem: int = 7,
    freeze_encoder: bool = True,
    freeze_decoder: bool = False,
    multimask: bool = True,
    normalization_mode: str = "0_255",
) -> SAMVTP:
    model = SAMVTP(
        sam2_cfg_path=str(resolve_path(sam2_cfg_path)),
        sam2_checkpoint_path=None,
        freeze_encoder=freeze_encoder,
        freeze_decoder=freeze_decoder,
        num_maskmem=num_maskmem,
        multimask=multimask,
    ).to(device)

    if sam_vtp_checkpoint_path:
        state = load_checkpoint_state(sam_vtp_checkpoint_path, map_location="cpu")
        model.load_state_dict(state, strict=False)

    model.input_normalization_mode = normalization_mode
    model.eval()
    return model
