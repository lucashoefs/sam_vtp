"""Optional CUDA-graph replay of the whole forward pass.

Rationale
---------
SAM-VTP is launch-bound at inference: the per-frame work is many small kernels,
so on a fast GPU (and even more so on an embedded one, where the CPU issuing the
launches is weak) the launch overhead is a large share of frame time. Capturing
the forward pass once and replaying it removes that overhead.

The obstacle is that the model is recurrent through a Python dict, which a graph
cannot walk. So this runner mirrors the memory read path over *fixed tensors*:
the bank becomes a stacked tensor, eviction becomes a shift-and-copy, the frame
counter becomes a device scalar, and the stored per-frame poses become another
stacked tensor. Once the bank is full the control flow is constant and the whole
pass is capturable, geometry and ego-motion warp included.

Everything the geometry path does is fixed-shape given a fixed input size: the
derived channels are interpolate, median, meshgrid, cross product and two box
convolutions, and the warp is a single `grid_sample` over a constant 7-fold
batch. The current frame's depth and pose are static input buffers, exactly like
the image.

Limitations
-----------
  * `memory_write_every > 1` disables it: the capture bakes in the write.
  * The bank must be full before capture, which `min_history_frames` enforces.
    Warmup frames run eagerly.
  * This is a second implementation of the forward pass. Run
    `scripts/verify_fullgraph_parity.py` after changing `architecture.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .architecture import (
    SAMVTP,
    downsample_depth,
    geometry_channels,
    get_1d_sine_pe,
    warp_features,
)


@dataclass
class _TensorState:
    """The recurrent state, as fixed-shape tensors a graph can update in place."""

    cond_maskmem_features: torch.Tensor      # (B, mem_dim, Hm, Wm)
    cond_maskmem_pos_enc: torch.Tensor       # (B, mem_dim, Hm, Wm)
    cond_obj_ptr: torch.Tensor               # (B, C)
    cond_pose: torch.Tensor                  # (B, 7)
    non_cond_maskmem_features: torch.Tensor  # (k, B, mem_dim, Hm, Wm)
    non_cond_maskmem_pos_enc: torch.Tensor   # (k, B, mem_dim, Hm, Wm)
    non_cond_obj_ptr: torch.Tensor           # (k, B, C)
    non_cond_pose: torch.Tensor              # (k, B, 7)
    step_idx: torch.Tensor


class SAMVTPFullGraphReplayRunner:
    """Full-forward CUDA graph replay with tensorised recurrent state."""

    def __init__(self, model: SAMVTP, device: torch.device, enabled: bool,
                 capture_warmup: int = 0, min_history_frames: Optional[int] = None):
        self.model = model
        self.device = device
        self.enabled = (bool(enabled) and device.type == "cuda"
                        and hasattr(torch.cuda, "CUDAGraph"))

        self._cadence_conflict = int(getattr(model, "memory_write_every", 1)) > 1
        if self.enabled and self._cadence_conflict:
            self.enabled = False

        # With memory switched off there is no recurrent state to tensorise: the
        # bank never fills, so capture could never trigger and every frame would
        # quietly fall through to eager. Say so instead of reporting "capture
        # pending" forever. The frame-wise path is cheap anyway -- it skips
        # memory attention and the memory encoder entirely.
        self._framewise = not bool(getattr(model, "use_memory", True))
        if self.enabled and self._framewise:
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
        # Component configuration the graph was captured under; compared on every
        # replay so a later switch change cannot go unnoticed.
        self._captured_component_epoch = -1
        self._capture_failed = False
        self._capture_error = ""
        self._graph = None
        self._state: Optional[_TensorState] = None
        self._static_input: Optional[torch.Tensor] = None
        self._static_depth: Optional[torch.Tensor] = None
        self._static_pose: Optional[torch.Tensor] = None
        self._static_valid: Optional[torch.Tensor] = None
        self._static_output: Optional[Dict[str, torch.Tensor]] = None

        self._k = max(0, int(model.num_maskmem) - 1)
        self._obj_ptr_tdiff_max = max(1.0, float(model.sam2_model.max_obj_ptrs_in_encoder - 1))
        self._noncond_tdiff = torch.arange(1, self._k + 1, device=device, dtype=torch.float32)

    def status_note(self) -> str:
        if not self.enabled:
            if self._cadence_conflict:
                return (f"CUDA Graph disabled (memory_write_every="
                        f"{self.model.memory_write_every} needs eager control flow)")
            if self._framewise:
                return "CUDA Graph disabled (memory switched off: no recurrent state to capture)"
            return "CUDA Graph disabled"
        if self._capture_failed:
            return f"CUDA Graph fell back to eager (capture failed: {self._capture_error})"
        if self._captured:
            extra = []
            if self.model.warp_memory:
                extra.append("warp")
            if self.model.use_geometry:
                extra.append("geometry")
            suffix = f", {' + '.join(extra)} in graph" if extra else ""
            return f"CUDA Graph replay active (full forward{suffix})"
        return "CUDA Graph enabled (history warmup / capture pending)"

    @staticmethod
    def _graph_context(graph):
        try:
            return torch.cuda.graph(graph, capture_error_mode="thread_local")
        except TypeError:  # older torch
            return torch.cuda.graph(graph)

    # -- state ---------------------------------------------------------------
    def _sync_tensor_state_from_model(self) -> bool:
        """Copy the model's dict-based memory into fixed tensors.

        Returns False until the bank is full, and (when the model warps) until
        every slot carries the pose it was observed from.
        """
        cond = self.model.output_dict.get("cond_frame_outputs", {})
        non_cond = self.model.output_dict.get("non_cond_frame_outputs", {})
        if (0 not in cond) or (self._k > 0 and len(non_cond) < self._k):
            return False

        needs_pose = bool(self.model.warp_memory)
        c = cond[0]
        if needs_pose and c.get("pose") is None:
            return False

        cond_feats = c["maskmem_features"].to(self.device)
        cond_pos = c["maskmem_pos_enc"][-1].to(self.device)
        cond_ptr = c["obj_ptr"].to(self.device)
        cond_pose = (c["pose"].to(self.device) if needs_pose
                     else cond_ptr.new_zeros(cond_ptr.shape[0], 7))

        if self._k > 0:
            items = sorted(non_cond.items(), key=lambda kv: kv[0])[-self._k :]
            if needs_pose and any(it[1].get("pose") is None for it in items):
                return False
            nc_feats = torch.stack([it[1]["maskmem_features"].to(self.device) for it in items], 0)
            nc_pos = torch.stack([it[1]["maskmem_pos_enc"][-1].to(self.device) for it in items], 0)
            nc_ptr = torch.stack([it[1]["obj_ptr"].to(self.device) for it in items], 0)
            nc_pose = (torch.stack([it[1]["pose"].to(self.device) for it in items], 0)
                       if needs_pose else nc_ptr.new_zeros(self._k, *cond_pose.shape))
        else:
            nc_feats = cond_feats.new_zeros((0, *cond_feats.shape))
            nc_pos = cond_pos.new_zeros((0, *cond_pos.shape))
            nc_ptr = cond_ptr.new_zeros((0, *cond_ptr.shape))
            nc_pose = cond_pose.new_zeros((0, *cond_pose.shape))

        self._state = _TensorState(
            cond_maskmem_features=cond_feats,
            cond_maskmem_pos_enc=cond_pos,
            cond_obj_ptr=cond_ptr,
            cond_pose=cond_pose,
            non_cond_maskmem_features=nc_feats,
            non_cond_maskmem_pos_enc=nc_pos,
            non_cond_obj_ptr=nc_ptr,
            non_cond_pose=nc_pose,
            step_idx=torch.tensor(float(self.model.frame_idx), device=self.device,
                                  dtype=torch.float32),
        )
        return True

    # -- forward -------------------------------------------------------------
    def _warped_bank(self, feat_size) -> torch.Tensor:
        """Every stored feature map resampled into the current camera frame.

        Returns (1 + k, B, mem_dim, Hm, Wm), conditioning slot first, matching
        the order `_memory_conditioned_pix_feat` consumes. All slots share the
        current depth and pose and differ only in their own stored pose, so the
        whole bank warps in one `grid_sample` over a constant batch.
        """
        s = self._state
        feats = torch.cat([s.cond_maskmem_features.unsqueeze(0),
                           s.non_cond_maskmem_features], dim=0)      # (S, B, Cm, Hm, Wm)
        poses = torch.cat([s.cond_pose.unsqueeze(0), s.non_cond_pose], dim=0)  # (S, B, 7)
        S, B = feats.shape[0], feats.shape[1]

        depth = downsample_depth(self._static_depth, tuple(feat_size))
        # reshape puts slot s, batch b at index s*B + b, and repeat(S, ...) lays
        # the current depth and pose out in exactly that order.
        flat = feats.reshape(S * B, *feats.shape[2:])
        warped, valid = warp_features(
            flat.float(),
            depth.float().repeat(S, 1, 1, 1),
            self._static_pose.float().repeat(S, 1),
            poses.reshape(S * B, 7).float(),
            mode=self.model.warp_mode, cam=self.model.camera,
            depth_valid=None if self._static_valid is None else
            downsample_depth(self._static_valid.float(), tuple(feat_size)).repeat(S, 1, 1, 1))

        warped = warped.to(feats.dtype)
        valid = valid.to(feats.dtype)
        blended = warped * valid + self.model.no_corr_embed.to(feats.dtype) * (1.0 - valid)
        return blended.reshape(S, B, *feats.shape[2:])

    def _memory_conditioned_pix_feat(self, vision_feats, vision_pos_embeds, feat_sizes):
        assert self._state is not None
        core = self.model.sam2_model
        B = vision_feats[-1].size(1)
        C = core.hidden_dim
        H, W = feat_sizes[-1]
        s = self._state

        if self.model.warp_memory and self._static_depth is not None:
            bank = self._warped_bank(feat_sizes[-1])
            cond_feat, non_cond_feat = bank[0], bank[1:]
        else:
            cond_feat, non_cond_feat = s.cond_maskmem_features, s.non_cond_maskmem_features

        to_cat = [cond_feat.flatten(2).permute(2, 0, 1)]
        to_cat_pos = [s.cond_maskmem_pos_enc.flatten(2).permute(2, 0, 1)
                      + core.maskmem_tpos_enc[self.model.num_maskmem - 1]]
        for t_pos in range(1, self.model.num_maskmem):
            slot = t_pos - 1
            to_cat.append(non_cond_feat[slot].flatten(2).permute(2, 0, 1))
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
        if model.use_geometry and self._static_depth is not None:
            top = backbone_out["backbone_fpn"][-1]
            geo = geometry_channels(self._static_depth,
                                    out_hw=(model.geometry_hw, model.geometry_hw),
                                    cam=model.camera, depth_valid=self._static_valid)
            backbone_out["backbone_fpn"] = list(backbone_out["backbone_fpn"])
            backbone_out["backbone_fpn"][-1] = top + model.geometry_encoder(
                geo.to(top.dtype), top.shape[-2:])

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
            if self.model.warp_memory:
                # The slot stores the pose it was observed from, so a later
                # frame can warp it into its own view.
                s.non_cond_pose[:-1].copy_(s.non_cond_pose[1:].clone())
                s.non_cond_pose[-1].copy_(self._static_pose)
        self._state.step_idx.add_(1.0)

        return {"logits": mask_logits}

    # -- capture -------------------------------------------------------------
    def _capture(self, frame: torch.Tensor, depth: Optional[torch.Tensor],
                 pose: Optional[torch.Tensor],
                 depth_valid: Optional[torch.Tensor]) -> None:
        s = self._state
        assert s is not None
        snap = (s.non_cond_maskmem_features.clone(), s.non_cond_maskmem_pos_enc.clone(),
                s.non_cond_obj_ptr.clone(), s.non_cond_pose.clone(), s.step_idx.clone())

        self._static_input = torch.empty_like(frame)
        self._static_depth = torch.empty_like(depth) if depth is not None else None
        self._static_pose = torch.empty_like(pose) if pose is not None else None
        self._static_valid = (torch.empty_like(depth_valid, dtype=torch.float32)
                              if depth_valid is not None else None)

        capture_stream = torch.cuda.Stream(device=self.device)
        torch.cuda.synchronize(self.device)
        with torch.cuda.stream(capture_stream):
            self._copy_inputs(frame, depth, pose, depth_valid)
            # Warmup populates the position-encoding and RoPE caches, which would
            # otherwise be built (and stored in Python dicts) mid-capture.
            for _ in range(self.capture_warmup):
                self._forward_tensorized_state(self._static_input)
            self._graph = torch.cuda.CUDAGraph()
            with self._graph_context(self._graph):
                self._static_output = self._forward_tensorized_state(self._static_input)
        torch.cuda.current_stream(self.device).wait_stream(capture_stream)

        s.non_cond_maskmem_features.copy_(snap[0])
        s.non_cond_maskmem_pos_enc.copy_(snap[1])
        s.non_cond_obj_ptr.copy_(snap[2])
        s.non_cond_pose.copy_(snap[3])
        s.step_idx.copy_(snap[4])
        self._captured = True
        self._captured_component_epoch = self.model.component_epoch

    def _copy_inputs(self, frame, depth, pose, depth_valid) -> None:
        self._static_input.copy_(frame)
        if self._static_depth is not None and depth is not None:
            self._static_depth.copy_(depth)
        if self._static_pose is not None and pose is not None:
            self._static_pose.copy_(pose)
        if self._static_valid is not None and depth_valid is not None:
            self._static_valid.copy_(depth_valid.float())

    def run(self, frame: torch.Tensor, is_init_frame: bool, update_memory: bool = True,
            depth: Optional[torch.Tensor] = None, pose: Optional[torch.Tensor] = None,
            depth_valid: Optional[torch.Tensor] = None):
        """Run one frame, replaying the captured graph when possible.

        `update_memory=False` is honoured only on the eager path, since a
        captured graph has the write baked into its control flow. The
        constructor already disables graph mode when the model carries a write
        cadence.
        """
        eager = lambda: self.model.forward_frame(  # noqa: E731
            frame, is_init_frame=is_init_frame, update_memory=update_memory,
            depth=depth, pose=pose, depth_valid=depth_valid)

        # A captured graph has warp/geometry/refine/memory baked into its control
        # flow, so a switch flipped afterwards would be ignored and the graph
        # would keep replaying the configuration it was captured with -- silently
        # reporting ablation numbers that are really the full model's, or the
        # reverse. Refuse instead.
        if self._captured and self.model.component_epoch != self._captured_component_epoch:
            raise RuntimeError(
                "component switches changed after the CUDA graph was captured; the "
                "graph still holds the old configuration. Call set_components() "
                "before the first frame, or build a new runner for the new "
                "configuration.")

        if (not self.enabled) or self._capture_failed:
            return eager()

        if is_init_frame or self._state is None:
            out = eager()
            if self.model.frame_idx >= self.min_history_frames:
                self._sync_tensor_state_from_model()
            return out

        if not self._captured:
            try:
                self._capture(frame, depth, pose, depth_valid)
            except Exception as exc:
                self._capture_failed = True
                self._capture_error = f"{type(exc).__name__}: {exc}"
                try:
                    torch.cuda.synchronize(self.device)
                except Exception:
                    pass
                return eager()

        self._copy_inputs(frame, depth, pose, depth_valid)
        self._graph.replay()
        return self._static_output
