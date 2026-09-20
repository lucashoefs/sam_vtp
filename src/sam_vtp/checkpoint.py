from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

BACKBONE_STRIDE = 16


def resolve_path(path_like) -> Path:
    return Path(path_like).expanduser().resolve()


def load_checkpoint_state(path, map_location="cpu") -> Dict[str, torch.Tensor]:
    """Just the weights, for callers that do not care about training metadata."""
    return read_checkpoint(path, map_location=map_location)[0]


def read_checkpoint(
    path, map_location="cpu"
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Return (state_dict, train_config).

    Handles the three nestings a SAM-VTP checkpoint may use: a bare state dict,
    or one under 'model_state_dict' or 'model'.
    """
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")

    raw = torch.load(p, map_location=map_location, weights_only=False)
    if not isinstance(raw, dict):
        raise TypeError(f"expected a dict checkpoint, got {type(raw).__name__}")

    state = raw.get("model_state_dict", raw.get("model", raw))
    cfg = raw.get("config", {})
    return state, cfg if isinstance(cfg, dict) else {}


def warp_state(state: Dict[str, torch.Tensor]) -> str:
    """absent | inert | trained."""
    t = state.get("no_corr_embed")
    if t is None:
        return "absent"
    return "inert" if bool(torch.all(t == 0)) else "trained"


def strip_inert_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Drop no_corr_embed when it is an all-zero leftover.

    The ego-motion memory warp was removed; checkpoints trained before that
    still carry its embedding. A zero one contributes nothing and the model has
    no parameter for it, which is why the target key count is 482 and not 483.
    A NON-zero one is a different model and must not be silently dropped -- see
    overrides_from_state.
    """
    if warp_state(state) == "inert":
        state = {k: v for k, v in state.items() if k != "no_corr_embed"}
    return state


def overrides_from_state(
    state: Dict[str, torch.Tensor], train_config: Dict[str, Any] | None = None
) -> List[Tuple[str, Any]]:
    """(dotted_path, value) pairs, in an order safe to apply sequentially."""
    if warp_state(state) == "trained":
        raise RuntimeError(
            "this checkpoint carries a NON-ZERO 'no_corr_embed': it was trained "
            "with the ego-motion memory warp, which this build does not "
            "implement. Loading it would score a model nobody trained."
        )

    def has(prefix: str) -> bool:
        return any(k.startswith(prefix) for k in state)

    out: List[Tuple[str, Any]] = []

    # --- sam2_model scalars ---------------------------------------------
    if "sam2_model.maskmem_tpos_enc" in state:
        out.append(
            (
                "model.sam2_model.num_maskmem",
                int(state["sam2_model.maskmem_tpos_enc"].shape[0]),
            )
        )

    dense = state.get("sam2_model.sam_prompt_encoder.dense_emb")
    if dense is not None:
        # image_size defines the PROMPT GRID, not the input resolution. A
        # 64x64 dense_emb means image_size 1024 even though inference runs at
        # 512; _prompt_tensors resamples the difference.
        out.append(
            ("model.sam2_model.image_size", int(dense.shape[-1]) * BACKBONE_STRIDE)
        )
    sparse = state.get("sam2_model.sam_prompt_encoder.sparse_tokens")
    if sparse is not None:
        out.append(("model.sam2_model.num_prompt_tokens", int(sparse.shape[0])))

    if train_config and "multimask_output" in train_config:
        out.append(
            (
                "model.sam2_model.multimask_output_in_sam",
                str(train_config["multimask_output"]) == "True",
            )
        )
        out.append(
            (
                "model.sam2_model.multimask_output_for_tracking",
                str(train_config["multimask_output"]) == "True",
            )
        )

    # --- components: scalar first, then the None that blanks the parent ---
    if has("refine_head."):
        out.append(
            (
                "model.refine_head.channels",
                int(state["refine_head.out.weight"].shape[1]),
            )
        )
    else:
        out.append(("model.refine_head", None))

    if not has("geometry_encoder."):
        out.append(("model.geometry_encoder", None))

    if not has("sam2_model.memory_attention."):
        # NOTE: SAM2Base creates maskmem_tpos_enc, no_mem_embed and
        # no_mem_pos_enc unconditionally (sam2_base.py:133-141), even with
        # memory_attention=None. A genuine SAM-TP checkpoint that lacks those
        # three will therefore report them as unexpected keys. Handle it when
        # you first load one.
        out.append(("model.sam2_model.memory_attention", None))
        out.append(("model.sam2_model.memory_encoder", None))

    return out
