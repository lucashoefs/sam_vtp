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
counter becomes a device scalar, and the stored per-frame features become another
stacked tensor. Once the bank is full the control flow is constant and the whole
pass is capturable, geometry included.

With memory switched off (the frame-wise ablation) none of that applies: there is
no recurrent state, so the pass is already pure feedforward and is captured
directly. That arm is supported so the ablation can be timed in the same regime
as the full model -- see `_framewise` in `__init__` for why the alternative
silently favours the memory pathway.

Everything the geometry path does is fixed-shape given a fixed input size: the
derived channels are interpolate, median, meshgrid, cross product and two box
convolutions. The current frame's depth is a static input buffer, exactly like
the image.

Write cadence
-------------
A frame that does not write to the bank differs from one that does only in the
tail of the pass: no memory encode, no shift, no step advance. That is a branch,
and a graph cannot take a branch -- but it can be two graphs, with the branch in
Python outside both. `capture_nowrite=True` captures the second one, which is
what lets a write cadence coexist with replay instead of turning it off.

The two graphs deliberately do NOT share a memory pool. Sharing one is the
documented way to save memory across captures, but it also lets a replay of one
graph land on the other's buffers; here the second pass costs a few hundred MB
once, which is the cheaper side of that trade on a device with memory to spare.

Limitations
-----------
  * With memory on, the bank must be full before capture, which
    `min_history_frames` enforces. Warmup frames run eagerly. With memory off
    there is no bank, so warmup only has to populate the lazy caches.
  * Once capture has happened there is no going back to eager: the dict-based
    memory in `architecture.py` stops being updated, so an eager frame after a
    replayed one would read stale state. Mismatches raise instead.
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
    geometry_channels,
    get_1d_sine_pe,
)


@dataclass
class _TensorState:
    """The recurrent state, as fixed-shape tensors a graph can update in place."""

    cond_maskmem_features: torch.Tensor      # (B, mem_dim, Hm, Wm)
    cond_maskmem_pos_enc: torch.Tensor       # (B, mem_dim, Hm, Wm)
    cond_obj_ptr: torch.Tensor               # (B, C)
    non_cond_maskmem_features: torch.Tensor  # (k, B, mem_dim, Hm, Wm)
    non_cond_maskmem_pos_enc: torch.Tensor   # (k, B, mem_dim, Hm, Wm)
    non_cond_obj_ptr: torch.Tensor           # (k, B, C)
    step_idx: torch.Tensor


class SAMVTPFullGraphReplayRunner:
    """Full-forward CUDA graph replay with tensorised recurrent state."""

    def __init__(self, model: SAMVTP, device: torch.device, enabled: bool,
                 capture_warmup: int = 0, min_history_frames: Optional[int] = None,
                 capture_nowrite: Optional[bool] = None):
        self.model = model
        self.device = device
        self.enabled = (bool(enabled) and device.type == "cuda"
                        and hasattr(torch.cuda, "CUDAGraph"))

        # A second graph for frames that skip the memory write. Requested
        # explicitly by a caller that gates on wall-clock time; inferred from the
        # model when the cadence is a fixed frame count.
        self._capture_nowrite = (int(getattr(model, "memory_write_every", 1)) > 1
                                 if capture_nowrite is None else bool(capture_nowrite))

        # With memory switched off there is no recurrent state to tensorise, so
        # the bank-filling handshake does not apply: the pass is pure feedforward
        # and capturable as soon as the lazy caches are warm.
        #
        # This arm used to be refused outright, which quietly biased any
        # comparison built on it -- timing the full model under replay and the
        # frame-wise ablation eagerly charges the entire replay speedup to the
        # frame-wise arm, making the memory pathway look cheaper than it is.
        # Both regimes have to be available for that comparison to mean anything.
        self._framewise = not bool(getattr(model, "use_memory", True))

        self.capture_warmup = max(0, int(capture_warmup))

        # The object-pointer temporal encoding is normalised by
        # min(frames_seen, max_obj_ptrs_in_encoder) - 1 in the eager path, so the
        # constant used below is only correct once enough frames have gone by.
        # That is why the floor is 16 and not num_maskmem.
        # Neither applies with memory off: there is no bank and no object-ptr
        # encoding, so frame-wise only needs enough eager frames to populate the
        # lazy position-encoding caches before capture.
        default_min_history = (2 if self._framewise else
                               max(int(model.num_maskmem),
                                   int(model.sam2_model.max_obj_ptrs_in_encoder)))
        self.min_history_frames = max(
            default_min_history,
            int(min_history_frames) if min_history_frames is not None else default_min_history)

        self._captured = False
        # Frame-wise readiness. Memory mode signals readiness by `_state` being
        # built; frame-wise has no state, so it counts warm eager frames instead.
        self._framewise_ready = False
        # Component configuration the graph was captured under; compared on every
        # replay so a later switch change cannot go unnoticed.
        self._captured_component_epoch = -1
        self._capture_failed = False
        self._capture_error = ""
        self._graph = None
        self._graph_nowrite = None
        self._state: Optional[_TensorState] = None
        self._static_input: Optional[torch.Tensor] = None
        self._static_depth: Optional[torch.Tensor] = None
        self._static_valid: Optional[torch.Tensor] = None
        self._static_output: Optional[Dict[str, torch.Tensor]] = None
        self._static_output_nowrite: Optional[Dict[str, torch.Tensor]] = None

        self._k = max(0, int(model.num_maskmem) - 1)
        self._obj_ptr_tdiff_max = max(1.0, float(model.sam2_model.max_obj_ptrs_in_encoder - 1))
        self._noncond_tdiff = torch.arange(1, self._k + 1, device=device, dtype=torch.float32)

    def status_note(self) -> str:
        if not self.enabled:
            return "CUDA Graph disabled"
        if self._capture_failed:
            return f"CUDA Graph fell back to eager (capture failed: {self._capture_error})"
        if self._captured:
            extra = []
            if self.model.use_geometry:
                extra.append("geometry")
            suffix = f", {' + '.join(extra)} in graph" if extra else ""
            cadence = ""
            if self._graph_nowrite is not None:
                every = int(getattr(self.model, "memory_write_every", 1))
                cadence = (f", 2 graphs (write every {every})" if every > 1
                           else ", 2 graphs (gated writes)")
            mode = "frame-wise" if self._framewise else "full forward"
            return f"CUDA Graph replay active ({mode}{suffix}{cadence})"
        return "CUDA Graph enabled (history warmup / capture pending)"

    def _capture_ready(self) -> bool:
        """Whether there is enough warm state to capture from."""
        if self._framewise:
            return self._framewise_ready
        return self._state is not None

    @staticmethod
    def _graph_context(graph):
        try:
            return torch.cuda.graph(graph, capture_error_mode="thread_local")
        except TypeError:  # older torch
            return torch.cuda.graph(graph)

    # -- state ---------------------------------------------------------------
    def _sync_tensor_state_from_model(self) -> bool:
        """Copy the model's dict-based memory into fixed tensors.

        Returns False until the bank is full.
        """
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

    def _forward_tensorized_state(self, frame: torch.Tensor,
                                  update_memory: bool = True) -> Dict[str, torch.Tensor]:
        assert self._framewise or self._state is not None
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
        if self._framewise:
            # Mirrors the no-memory branch in architecture.forward_frame, which
            # writes the token back into vision_feats[-1] rather than only into
            # pix_feat. Nothing downstream reads vision_feats[-1] on this path,
            # but keeping the two implementations identical is what makes
            # verify_fullgraph_parity meaningful.
            vision_feats[-1] = vision_feats[-1] + core.no_mem_embed
            pix_feat = vision_feats[-1].permute(1, 2, 0).view(B, -1, *feat_sizes[-1])
        else:
            pix_feat = self._memory_conditioned_pix_feat(
                vision_feats, vision_pos_embeds, feat_sizes)

        sparse, dense, image_pe = model._prompt_tensors(pix_feat, B)

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
        if model.refine_head is not None and model.use_refine_head:
            # Must run here as well: without it, graph mode returns unrefined
            # logits and eager mode returns refined ones, so the same checkpoint
            # gives two different answers depending on a speed flag. The
            # `use_refine_head` switch has to be honoured for the same reason --
            # omitting it made `--no-refine` a no-op under graph replay.
            mask_logits = model.refine_head(mask_logits, image_norm)

        if not update_memory:
            # Stop exactly where the eager path stops for a non-write frame
            # (architecture.forward_frame, `if not update_memory: return`):
            # no memory encode, no bank shift, and no step_idx advance, since
            # eager also returns before incrementing frame_idx.
            return {"logits": mask_logits}

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
    def _capture(self, frame: torch.Tensor, depth: Optional[torch.Tensor],
                 depth_valid: Optional[torch.Tensor]) -> None:
        s = self._state
        assert self._framewise or s is not None
        # Frame-wise has no bank, so there is nothing to snapshot or restore.
        writes = not self._framewise
        snap = None if s is None else (
            s.non_cond_maskmem_features.clone(), s.non_cond_maskmem_pos_enc.clone(),
            s.non_cond_obj_ptr.clone(), s.step_idx.clone())

        self._static_input = torch.empty_like(frame)
        self._static_depth = torch.empty_like(depth) if depth is not None else None
        self._static_valid = (torch.empty_like(depth_valid, dtype=torch.float32)
                              if depth_valid is not None else None)

        capture_stream = torch.cuda.Stream(device=self.device)
        torch.cuda.synchronize(self.device)
        with torch.cuda.stream(capture_stream):
            self._copy_inputs(frame, depth, depth_valid)
            # Warmup populates the position-encoding and RoPE caches, which would
            # otherwise be built (and stored in Python dicts) mid-capture.
            for _ in range(self.capture_warmup):
                self._forward_tensorized_state(self._static_input, update_memory=writes)
            self._graph = torch.cuda.CUDAGraph()
            with self._graph_context(self._graph):
                self._static_output = self._forward_tensorized_state(
                    self._static_input, update_memory=writes)

            # A frame-wise pass never writes, so the single graph above already
            # is the non-writing one; a second capture would be a duplicate.
            if self._capture_nowrite and not self._framewise:
                # Capturing this one *after* the write graph is deliberate: the
                # write graph's capture leaves the bank shifted, and this pass
                # only reads it, so whatever it sees is representative of a
                # steady-state frame. Own pool, so replaying one graph can never
                # scribble on the other's buffers.
                self._graph_nowrite = torch.cuda.CUDAGraph()
                with self._graph_context(self._graph_nowrite):
                    self._static_output_nowrite = self._forward_tensorized_state(
                        self._static_input, update_memory=False)
        torch.cuda.current_stream(self.device).wait_stream(capture_stream)

        if snap is not None:
            s.non_cond_maskmem_features.copy_(snap[0])
            s.non_cond_maskmem_pos_enc.copy_(snap[1])
            s.non_cond_obj_ptr.copy_(snap[2])
            s.step_idx.copy_(snap[3])
        self._captured = True
        self._captured_component_epoch = self.model.component_epoch

    def _copy_inputs(self, frame, depth, depth_valid) -> None:
        self._static_input.copy_(frame)
        if self._static_depth is not None and depth is not None:
            self._static_depth.copy_(depth)
        if self._static_valid is not None and depth_valid is not None:
            self._static_valid.copy_(depth_valid.float())

    def _writes_this_frame(self, update_memory: bool) -> bool:
        """Mirror the eager cadence gate, and advance the counter it reads.

        `architecture.forward_frame` decides on `_frames_seen % write_every` and
        then increments `_frames_seen`. Doing the same here on the *same*
        counter is what keeps the cadence in phase across the eager-to-replay
        handover: the graph path picking up its own counter at zero would
        restart the cadence and put the writes on different frames than the
        warmup had them on.
        """
        model = self.model
        if not model.use_memory:
            update_memory = False
        every = int(getattr(model, "memory_write_every", 1))
        if update_memory and every > 1 and (model._frames_seen % every) != 0:
            update_memory = False
        model._frames_seen += 1
        return update_memory

    def run(self, frame: torch.Tensor, is_init_frame: bool, update_memory: bool = True,
            depth: Optional[torch.Tensor] = None,
            depth_valid: Optional[torch.Tensor] = None):
        """Run one frame, replaying the captured graph when possible.

        `update_memory=False` is honoured under replay too, via the second
        captured graph. It needs `capture_nowrite=True` at construction; without
        it there is no non-writing graph to replay and the call raises, because
        silently writing anyway would corrupt the bank's temporal spacing and
        silently falling back to eager would read stale dict state.
        """
        eager = lambda: self.model.forward_frame(  # noqa: E731
            frame, is_init_frame=is_init_frame, update_memory=update_memory,
            depth=depth, depth_valid=depth_valid)

        # A captured graph has geometry/refine/memory baked into its control
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

        if is_init_frame or not self._capture_ready():
            out = eager()
            if self._framewise:
                # `frame_idx` never advances with memory off -- the eager path
                # returns before incrementing it -- so warmth is counted on
                # `_frames_seen`, which does advance on every frame.
                self._framewise_ready = (
                    self.model._frames_seen >= self.min_history_frames)
            elif self.model.frame_idx >= self.min_history_frames:
                self._sync_tensor_state_from_model()
            return out

        if not self._captured:
            try:
                self._capture(frame, depth, depth_valid)
            except Exception as exc:
                self._capture_failed = True
                self._capture_error = f"{type(exc).__name__}: {exc}"
                try:
                    torch.cuda.synchronize(self.device)
                except Exception:
                    pass
                return eager()

        writes = self._writes_this_frame(update_memory)
        if self._framewise:
            # Still called above so `_frames_seen` advances in step with eager,
            # but there is only one graph and it never writes, so the cadence
            # result carries no branch here.
            self._copy_inputs(frame, depth, depth_valid)
            self._graph.replay()
            return self._static_output
        if not writes and self._graph_nowrite is None:
            raise RuntimeError(
                "update_memory=False under CUDA-graph replay, but no non-writing "
                "graph was captured. Build the runner with capture_nowrite=True "
                "(SAMVTPStreamProcessor does this when a write cadence is set).")

        self._copy_inputs(frame, depth, depth_valid)
        if writes:
            self._graph.replay()
            return self._static_output
        self._graph_nowrite.replay()
        return self._static_output_nowrite
