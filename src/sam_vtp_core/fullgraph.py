"""Optional CUDA-graph replay of the whole forward pass.

Rationale
---------
SAM-VTP is launch-bound at inference: the per-frame work is many small kernels,
so on a fast GPU the Python and launch overhead is a large share of the frame
time. Capturing the forward pass once and replaying it removes that overhead.

The obstacle is that the model is recurrent through a Python dict, which a graph
cannot walk. So this runner mirrors the memory read path over *fixed tensors*:
the bank becomes a stacked tensor, eviction becomes a shift-and-copy, and the
frame counter becomes a device scalar. Once the bank is full, the control flow
is constant and the whole pass is capturable.

Limitations
-----------
  * RGB-only models. The ego-motion warp reads a variable set of stored poses
    and resamples against the current depth, which is not constant control
    flow, so a geometry checkpoint disables graph mode rather than running
    unwarped. That is the released configuration, so in practice this path
    serves ablations and SAM-TP-style baselines.
  * `memory_write_every > 1` disables it too: the capture bakes in the write.
  * This is a second implementation of the forward pass. Run
    `scripts/verify_fullgraph_parity.py` after changing `architecture.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .architecture import SAMVTP, get_1d_sine_pe


@dataclass
class _TensorState:
    cond_maskmem_features: torch.Tensor
    cond_maskmem_pos_enc: torch.Tensor
    cond_obj_ptr: torch.Tensor
    non_cond_maskmem_features: torch.Tensor
    non_cond_maskmem_pos_enc: torch.Tensor
    non_cond_obj_ptr: torch.Tensor
    step_idx: torch.Tensor


class SAMVTPFullGraphReplayRunner:
    """Full-forward CUDA graph replay with tensorised recurrent state."""

    def __init__(self, model: SAMVTP, device: torch.device, enabled: bool,
                 capture_warmup: int = 0, min_history_frames: Optional[int] = None):
        self.model = model
        self.device = device
        self.enabled = (bool(enabled) and device.type == "cuda"
                        and hasattr(torch.cuda, "CUDAGraph"))

        self._geometry_conflict = bool(model.needs_geometry)
        self._cadence_conflict = int(getattr(model, "memory_write_every", 1)) > 1
        if self.enabled and (self._geometry_conflict or self._cadence_conflict):
            self.enabled = False

        self.capture_warmup = max(0, int(capture_warmup))

        # The object-pointer temporal encoding is normalised by
        # min(frames_seen, max_obj_ptrs_in_encoder) - 1 in the eager path, so the
        # constant used below is only correct once enough frames have gone by.
        # That is why the floor is 16 and not num_maskmem.
        default_min_history = max(int(model.num_maskmem),
                                  int(model.sam2_model.max_obj_ptrs_in_encoder))
        self.min_history_frames = max(
            default_min_history,
            int(min_history_frames) if min_history_frames is not None else default_min_history)

        self._captured = False
        self._capture_failed = False
        self._capture_error = ""
        self._graph = None
        self._state: Optional[_TensorState] = None
        self._static_input: Optional[torch.Tensor] = None
        self._static_output: Optional[Dict[str, torch.Tensor]] = None

        self._k = max(0, int(model.num_maskmem) - 1)
        self._obj_ptr_tdiff_max = max(1.0, float(model.sam2_model.max_obj_ptrs_in_encoder - 1))
        self._noncond_tdiff = torch.arange(1, self._k + 1, device=device, dtype=torch.float32)

    def status_note(self) -> str:
        if not self.enabled:
            if self._geometry_conflict:
                return ("CUDA Graph disabled (ego-motion warp / geometry input need "
                        "eager control flow)")
            if self._cadence_conflict:
                return (f"CUDA Graph disabled (memory_write_every="
                        f"{self.model.memory_write_every} needs eager control flow)")
            return "CUDA Graph disabled"
        if self._capture_failed:
            return f"CUDA Graph fell back to eager (capture failed: {self._capture_error})"
        if self._captured:
            return "CUDA Graph replay active (full forward, in-graph state update)"
        return "CUDA Graph enabled (history warmup / capture pending)"

    @staticmethod
    def _graph_context(graph):
        try:
            return torch.cuda.graph(graph, capture_error_mode="thread_local")
        except TypeError:  # older torch
            return torch.cuda.graph(graph)

    # -- state ---------------------------------------------------------------
    def _sync_tensor_state_from_model(self) -> bool:
        cond = self.model.output_dict.get("cond_frame_outputs", {})
        non_cond = self.model.output_dict.get("non_cond_frame_outputs", {})
        if (0 not in cond) or (self._k > 0 and len(non_cond) < self._k):
            return False

        c = cond[0]
        cond_feats = c["maskmem_features"].to(self.device)
        cond_pos = c["maskmem_pos_enc"][-1].to(self.device)
        cond_ptr = c["obj_ptr"].to(self.device)

        if self._k > 0:
            items = sorted(non_cond.items(), key=lambda kv: kv[0])[-self._k :]
            nc_feats = torch.stack([it[1]["maskmem_features"].to(self.device) for it in items], 0)
            nc_pos = torch.stack([it[1]["maskmem_pos_enc"][-1].to(self.device) for it in items], 0)
            nc_ptr = torch.stack([it[1]["obj_ptr"].to(self.device) for it in items], 0)
        else:
            nc_feats = cond_feats.new_zeros((0, *cond_feats.shape))
            nc_pos = cond_pos.new_zeros((0, *cond_pos.shape))
            nc_ptr = cond_ptr.new_zeros((0, *cond_ptr.shape))

        self._state = _TensorState(
            cond_maskmem_features=cond_feats,
            cond_maskmem_pos_enc=cond_pos,
            cond_obj_ptr=cond_ptr,
            non_cond_maskmem_features=nc_feats,
            non_cond_maskmem_pos_enc=nc_pos,
            non_cond_obj_ptr=nc_ptr,
            step_idx=torch.tensor(float(self.model.frame_idx), device=self.device,
                                  dtype=torch.float32),
        )
        return True

    # -- forward -------------------------------------------------------------
    def _memory_conditioned_pix_feat(self, vision_feats, vision_pos_embeds, feat_sizes):
        assert self._state is not None
        core = self.model.sam2_model
        B = vision_feats[-1].size(1)
        C = core.hidden_dim
        H, W = feat_sizes[-1]
        s = self._state

        to_cat = [s.cond_maskmem_features.flatten(2).permute(2, 0, 1)]
        to_cat_pos = [s.cond_maskmem_pos_enc.flatten(2).permute(2, 0, 1)
                      + core.maskmem_tpos_enc[self.model.num_maskmem - 1]]
        for t_pos in range(1, self.model.num_maskmem):
            slot = t_pos - 1
            to_cat.append(s.non_cond_maskmem_features[slot].flatten(2).permute(2, 0, 1))
            to_cat_pos.append(s.non_cond_maskmem_pos_enc[slot].flatten(2).permute(2, 0, 1)
                              + core.maskmem_tpos_enc[self.model.num_maskmem - t_pos - 1])

        obj_ptrs = torch.cat([s.cond_obj_ptr.unsqueeze(0),
                              torch.flip(s.non_cond_obj_ptr, dims=[0])], dim=0)
        tdiff = torch.cat([s.step_idx.view(1), self._noncond_tdiff], dim=0)
        obj_pos = get_1d_sine_pe(tdiff / self._obj_ptr_tdiff_max, dim=C)
        obj_pos = core.obj_ptr_tpos_proj(obj_pos).unsqueeze(1).expand(-1, B, core.mem_dim)
        if core.mem_dim < C:
            ptr_repeat = C // core.mem_dim
            obj_ptrs = obj_ptrs.reshape(-1, B, ptr_repeat, core.mem_dim)
            obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
            obj_pos = (obj_pos.unsqueeze(1).expand(-1, ptr_repeat, -1, -1)
                       .reshape(-1, B, core.mem_dim))
        to_cat.append(obj_ptrs)
        to_cat_pos.append(obj_pos)

        out = core.memory_attention(
            curr=vision_feats[-1], curr_pos=vision_pos_embeds[-1],
            memory=torch.cat(to_cat, dim=0), memory_pos=torch.cat(to_cat_pos, dim=0),
            num_obj_ptr_tokens=obj_ptrs.shape[0])
        return out.permute(1, 2, 0).view(B, C, H, W)

    def _forward_tensorized_state(self, frame: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert self._state is not None
        model, core = self.model, self.model.sam2_model
        B, _, H, W = frame.shape
        image_norm = model.normalize_imagenet(frame)

        backbone_out = core.forward_image(image_norm)
        vision_feats, vision_pos_embeds, feat_sizes = core.prepare_backbone_features(backbone_out)

        pix_feat = self._memory_conditioned_pix_feat(vision_feats, vision_pos_embeds, feat_sizes)

        sparse, dense = core.sam_prompt_encoder(points=None, boxes=None, masks=None)
        sparse = sparse.expand(B, -1, -1)
        dense = dense.expand(B, -1, -1, -1)
        if dense.shape[-2:] != pix_feat.shape[-2:]:
            dense = F.interpolate(dense, size=pix_feat.shape[-2:], mode="bilinear",
                                  align_corners=False)
        image_pe = core.sam_prompt_encoder.get_dense_pe()
        if image_pe.shape[-2:] != pix_feat.shape[-2:]:
            image_pe = F.interpolate(image_pe, size=pix_feat.shape[-2:], mode="bilinear",
                                     align_corners=False)
        image_pe = image_pe.to(device=pix_feat.device, dtype=pix_feat.dtype)

        high_res_feats = [f.permute(1, 2, 0).view(B, -1, *sz)
                          for f, sz in zip(vision_feats[:-1], feat_sizes[:-1])]

        low_res_masks, iou_preds, obj_ptr, _ = core.sam_mask_decoder(
            image_embeddings=pix_feat, image_pe=image_pe,
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=model.multimask_output, repeat_image=False,
            high_res_features=high_res_feats)

        if model.multimask_output:
            best = iou_preds.argmax(dim=1)
            rows = torch.arange(B, device=frame.device)
            mask_logits = low_res_masks[rows, best].unsqueeze(1)
            obj_ptr = obj_ptr.gather(1, best.view(B, 1, 1).expand(-1, 1, obj_ptr.shape[-1]))
        else:
            mask_logits = low_res_masks
        obj_ptr = obj_ptr.squeeze(1)

        scale = max(H, W) / max(mask_logits.shape[-2], mask_logits.shape[-1])
        mask_logits = F.interpolate(mask_logits, size=(H, W),
                                    mode="bicubic" if scale >= 3 else "bilinear",
                                    align_corners=False)
        if model.refine_head is not None:
            # Must run here as well: without it, graph mode returns unrefined
            # logits and eager mode returns refined ones, so the same checkpoint
            # gives two different answers depending on a speed flag.
            mask_logits = model.refine_head(mask_logits, image_norm)

        object_score_logits = torch.any(mask_logits.flatten(1) > 0.0, dim=1, keepdim=True).float()
        maskmem_features, maskmem_pos_enc = core.encode_new_memory(
            current_vision_feats=vision_feats, feat_sizes=feat_sizes,
            pred_masks_high_res=mask_logits, object_score_logits=object_score_logits)

        if self._k > 0:
            s = self._state
            s.non_cond_maskmem_features[:-1].copy_(s.non_cond_maskmem_features[1:].clone())
            s.non_cond_maskmem_pos_enc[:-1].copy_(s.non_cond_maskmem_pos_enc[1:].clone())
            s.non_cond_obj_ptr[:-1].copy_(s.non_cond_obj_ptr[1:].clone())
            s.non_cond_maskmem_features[-1].copy_(maskmem_features)
            s.non_cond_maskmem_pos_enc[-1].copy_(maskmem_pos_enc[-1])
            s.non_cond_obj_ptr[-1].copy_(obj_ptr)
        self._state.step_idx.add_(1.0)

        return {"logits": mask_logits}

    # -- capture -------------------------------------------------------------
    def _capture(self, frame: torch.Tensor) -> None:
        s = self._state
        assert s is not None
        snap = (s.non_cond_maskmem_features.clone(), s.non_cond_maskmem_pos_enc.clone(),
                s.non_cond_obj_ptr.clone(), s.step_idx.clone())

        self._static_input = torch.empty_like(frame)
        capture_stream = torch.cuda.Stream(device=self.device)
        torch.cuda.synchronize(self.device)
        with torch.cuda.stream(capture_stream):
            self._static_input.copy_(frame)
            for _ in range(self.capture_warmup):
                self._forward_tensorized_state(self._static_input)
            self._graph = torch.cuda.CUDAGraph()
            with self._graph_context(self._graph):
                self._static_output = self._forward_tensorized_state(self._static_input)
        torch.cuda.current_stream(self.device).wait_stream(capture_stream)

        s.non_cond_maskmem_features.copy_(snap[0])
        s.non_cond_maskmem_pos_enc.copy_(snap[1])
        s.non_cond_obj_ptr.copy_(snap[2])
        s.step_idx.copy_(snap[3])
        self._captured = True

    def run(self, frame: torch.Tensor, is_init_frame: bool, update_memory: bool = True,
            depth: Optional[torch.Tensor] = None, pose: Optional[torch.Tensor] = None):
        """Run one frame, replaying the captured graph when possible.

        `update_memory=False` is honoured only on the eager path -- a captured
        graph has the write baked into its control flow. The constructor already
        disables graph mode when the model carries a write cadence.
        """
        eager = lambda: self.model.forward_frame(  # noqa: E731
            frame, is_init_frame=is_init_frame, update_memory=update_memory,
            depth=depth, pose=pose)

        if (not self.enabled) or self._capture_failed:
            return eager()

        if is_init_frame or self._state is None:
            out = eager()
            if self.model.frame_idx >= self.min_history_frames:
                self._sync_tensor_state_from_model()
            return out

        if not self._captured:
            try:
                self._capture(frame)
            except Exception as exc:
                self._capture_failed = True
                self._capture_error = f"{type(exc).__name__}: {exc}"
                try:
                    torch.cuda.synchronize(self.device)
                except Exception:
                    pass
                return eager()

        self._static_input.copy_(frame)
        self._graph.replay()
        return self._static_output
