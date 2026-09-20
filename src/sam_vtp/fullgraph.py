"""CUDA-graph replay for the complete SAM-VTP forward pass.

The normal SAM-VTP forward pass keeps temporal memory in Python dictionaries.
CUDA graphs cannot follow that dynamic structure, so this runner converts the
recurrent state into fixed-size tensors and replays the whole frame pass from
those tensors.

There are two execution modes:

* Frame-wise mode (memory disabled): the forward pass is feed-forward, so one
  graph is enough.
* Full mode (memory enabled): the memory bank is tensorised and updated in the
  graph.  A second graph can optionally handle frames that skip the memory
  write.

The graph path intentionally mirrors ``SAMVTP.forward_frame``.  This file is
therefore an execution backend rather than a second model definition: changes
to the eager forward path should be checked against this implementation with
tests/check_fullgraph.py.

File Contents:
1. Status ................................................................ 134
2. Memory state .......................................................... 179
3. Forward
    3.1. encoder and geometry ............................................ 242
    3.2. temporal memory ................................................. 280
    3.3. decoder ......................................................... 389
    3.4. memory write .................................................... 449
    3.5. complete graph-compatible pass .................................. 496
4. Capture ............................................................... 550
5. Frame cadence ......................................................... 670
6. Execution ............................................................. 694
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from sam_vtp.modeling.geometry import geometry_channels
from sam_vtp.modeling.sam_vtp import SAMVTP
from sam_vtp.modeling.sam_vtp_base import get_1d_sine_pe


@dataclass
class GraphMemoryState:
    """Fixed-shape recurrent state that CUDA graph replay can update in place."""

    cond_maskmem_features: torch.Tensor         # (B, mem_dim, Hm, Wm)
    cond_maskmem_pos_enc: torch.Tensor          # (B, mem_dim, Hm, Wm)
    cond_obj_ptr: torch.Tensor                  # (B, C)
    non_cond_maskmem_features: torch.Tensor     # (k, B, mem_dim, Hm, Wm)
    non_cond_maskmem_pos_enc: torch.Tensor      # (k, B, mem_dim, Hm, Wm)
    non_cond_obj_ptr: torch.Tensor              # (k, B, C)
    step_idx: torch.Tensor                      # scalar float32 device tensor


class SAMVTPFullGraphReplayRunner:
    """Replay the complete SAM-VTP forward pass with CUDA graphs."""

    def __init__(
        self,
        model: SAMVTP,
        device: torch.device,
        enabled: bool,
        capture_warmup: int = 0,
        min_history_frames: Optional[int] = None,
        capture_nowrite: Optional[bool] = None,
    ):
        self.model = model
        self.device = device
        self.enabled = (
            bool(enabled) and device.type == "cuda" and hasattr(torch.cuda, "CUDAGraph")
        )

        # With a fixed write cadence, a writing frame and a non-writing frame
        # have different graph control flow. Capture a second graph for the
        # latter so the branch remains outside CUDA graph replay.
        self._capture_nowrite = (
            int(getattr(model, "memory_write_every", 1)) > 1
            if capture_nowrite is None
            else bool(capture_nowrite)
        )

        # Memory-off mode has no recurrent state to tensorise. It still uses
        # CUDA graph replay so its timing is comparable to the full model.
        self._framewise = not bool(getattr(model, "use_memory", True))
        self.capture_warmup = max(0, int(capture_warmup))

        # Object-pointer temporal encoding in the eager path eventually uses
        # max_obj_ptrs_in_encoder - 1 as its normalised time denominator. We
        # therefore wait until enough eager frames have passed before capture.
        # Frame-wise mode only needs enough frames to warm lazy caches.
        default_min_history = (
            2
            if self._framewise
            else max(
                int(model.num_maskmem),
                int(model.sam2_model.max_obj_ptrs_in_encoder),
            )
        )
        self.min_history_frames = max(
            default_min_history,
            (
                int(min_history_frames)
                if min_history_frames is not None
                else default_min_history
            ),
        )

        # Capture state.
        self._captured = False
        self._framewise_ready = False
        self._captured_component_epoch = -1
        self._capture_failed = False
        self._capture_error = ""

        # CUDA graph objects and their static buffers.
        self._graph = None
        self._graph_nowrite = None
        self._static_input: Optional[torch.Tensor] = None
        self._static_depth: Optional[torch.Tensor] = None
        self._static_valid: Optional[torch.Tensor] = None
        self._static_output: Optional[Dict[str, torch.Tensor]] = None
        self._static_output_nowrite: Optional[Dict[str, torch.Tensor]] = None

        # Fixed temporal memory layout.
        self._state: Optional[GraphMemoryState] = None
        self._k = max(0, int(model.num_maskmem) - 1)
        self._obj_ptr_tdiff_max = max(
            1.0,
            float(model.sam2_model.max_obj_ptrs_in_encoder - 1),
        )
        self._noncond_tdiff = torch.arange(
            1,
            self._k + 1,
            device=device,
            dtype=torch.float32,
        )

    # ---------------------------------------------------------------------
    # 1. Status / capture readiness
    # ---------------------------------------------------------------------

    def status_note(self) -> str:
        """Return a short human-readable description of runner state."""
        if not self.enabled:
            return "CUDA Graph disabled"

        if self._capture_failed:
            return (
                "CUDA Graph fell back to eager "
                f"(capture failed: {self._capture_error})"
            )

        if self._captured:
            suffix = ", geometry in graph" if self.model.use_geometry else ""
            cadence = ""
            if self._graph_nowrite is not None:
                every = int(getattr(self.model, "memory_write_every", 1))
                cadence = (
                    f", 2 graphs (write every {every})"
                    if every > 1
                    else ", 2 graphs (gated writes)"
                )
            mode = "frame-wise" if self._framewise else "full forward"
            return f"CUDA Graph replay active ({mode}{suffix}{cadence})"

        return "CUDA Graph enabled (history warmup / capture pending)"

    def _capture_ready(self) -> bool:
        """Whether enough eager state exists to capture the graph."""
        if self._framewise:
            return self._framewise_ready
        return self._state is not None

    @staticmethod
    def _graph_context(graph):
        """Use thread-local capture mode when the installed PyTorch supports it."""
        try:
            return torch.cuda.graph(graph, capture_error_mode="thread_local")
        except TypeError:  # older torch
            return torch.cuda.graph(graph)

    # ---------------------------------------------------------------------
    # 2. Memory state
    # ---------------------------------------------------------------------

    def _memory_bank_is_ready(self) -> bool:
        """Return whether the eager model has enough history to build a bank."""
        cond = self.model.output_dict.get("cond_frame_outputs", {})
        non_cond = self.model.output_dict.get("non_cond_frame_outputs", {})
        return 0 in cond and (self._k == 0 or len(non_cond) >= self._k)

    def _build_graph_memory_state(self) -> GraphMemoryState:
        """Convert SAM2's dictionary memory into fixed-shape graph tensors."""
        cond = self.model.output_dict["cond_frame_outputs"][0]
        non_cond = self.model.output_dict.get("non_cond_frame_outputs", {})

        cond_features = cond["maskmem_features"].to(self.device)
        cond_pos = cond["maskmem_pos_enc"][-1].to(self.device)
        cond_ptr = cond["obj_ptr"].to(self.device)

        if self._k > 0:
            # Keep the most recent non-conditioning frames in chronological
            # order.  This is the same history that SAM2's dict path exposes.
            items = sorted(non_cond.items(), key=lambda item: item[0])[-self._k :]
            non_cond_features = torch.stack(
                [item[1]["maskmem_features"].to(self.device) for item in items],
                dim=0,
            )
            non_cond_pos = torch.stack(
                [item[1]["maskmem_pos_enc"][-1].to(self.device) for item in items],
                dim=0,
            )
            non_cond_ptr = torch.stack(
                [item[1]["obj_ptr"].to(self.device) for item in items],
                dim=0,
            )
        else:
            # Keep the zero-history case shape-compatible with the general path.
            non_cond_features = cond_features.new_zeros((0, *cond_features.shape))
            non_cond_pos = cond_pos.new_zeros((0, *cond_pos.shape))
            non_cond_ptr = cond_ptr.new_zeros((0, *cond_ptr.shape))

        return GraphMemoryState(
            cond_maskmem_features=cond_features,
            cond_maskmem_pos_enc=cond_pos,
            cond_obj_ptr=cond_ptr,
            non_cond_maskmem_features=non_cond_features,
            non_cond_maskmem_pos_enc=non_cond_pos,
            non_cond_obj_ptr=non_cond_ptr,
            step_idx=torch.tensor(
                float(self.model.frame_idx),
                device=self.device,
                dtype=torch.float32,
            ),
        )

    def _sync_tensor_state_from_model(self) -> bool:
        """Build the fixed graph memory state once the eager bank is full."""
        if not self._memory_bank_is_ready():
            return False
        self._state = self._build_graph_memory_state()
        return True

    # ---------------------------------------------------------------------
    # 3.1. Forward: encoder and geometry
    # ---------------------------------------------------------------------

    def _encode_backbone(
        self,
        frame: torch.Tensor,
        depth: Optional[torch.Tensor],
        depth_valid: Optional[torch.Tensor],
    ):
        """Run the image encoder and optionally inject geometry at its top level."""
        model = self.model
        core = model.sam2_model
        image_norm = model.normalize_imagenet(frame)
        backbone_out = core.forward_image(image_norm)

        if model.use_geometry and depth is not None:
            top = backbone_out["backbone_fpn"][-1]
            geo = geometry_channels(
                depth,
                out_hw=(model.geometry_hw, model.geometry_hw),
                cam=model.camera,
                depth_valid=depth_valid,
            )
            backbone_out["backbone_fpn"] = list(backbone_out["backbone_fpn"])
            backbone_out["backbone_fpn"][-1] = top + model.geometry_encoder(
                geo.to(top.dtype),
                top.shape[-2:],
            )

        # This SAM2Base version returns four values; the first is unused here.
        _, vision_feats, vision_pos_embeds, feat_sizes = (
            core._prepare_backbone_features(backbone_out)
        )
        return image_norm, vision_feats, vision_pos_embeds, feat_sizes

    # ---------------------------------------------------------------------
    # 3.2. Forward: temporal memory
    # ---------------------------------------------------------------------

    def _build_mask_memory(self):
        """Flatten the fixed mask-memory bank into memory-attention tokens."""
        assert self._state is not None
        core = self.model.sam2_model
        state = self._state

        memory = [state.cond_maskmem_features.flatten(2).permute(2, 0, 1)]
        memory_pos = [
            state.cond_maskmem_pos_enc.flatten(2).permute(2, 0, 1)
            + core.maskmem_tpos_enc[self.model.num_maskmem - 1]
        ]

        for t_pos in range(1, self.model.num_maskmem):
            slot = t_pos - 1
            memory.append(
                state.non_cond_maskmem_features[slot].flatten(2).permute(2, 0, 1)
            )
            memory_pos.append(
                state.non_cond_maskmem_pos_enc[slot].flatten(2).permute(2, 0, 1)
                + core.maskmem_tpos_enc[self.model.num_maskmem - t_pos - 1]
            )

        return torch.cat(memory, dim=0), torch.cat(memory_pos, dim=0)

    def _build_object_pointer_memory(self, batch_size: int):
        """Build temporal object-pointer tokens in SAM2's expected ordering."""
        assert self._state is not None
        core = self.model.sam2_model
        state = self._state
        hidden_dim = core.hidden_dim

        # Conditioning pointer comes first.  Non-conditioning pointers are
        # stored oldest -> newest, but SAM2 consumes them newest -> oldest.
        obj_ptrs = torch.cat(
            [
                state.cond_obj_ptr.unsqueeze(0),
                torch.flip(state.non_cond_obj_ptr, dims=[0]),
            ],
            dim=0,
        )

        tdiff = torch.cat(
            [state.step_idx.view(1), self._noncond_tdiff],
            dim=0,
        )
        obj_pos = get_1d_sine_pe(
            tdiff / self._obj_ptr_tdiff_max,
            dim=hidden_dim,
        )
        obj_pos = (
            core.obj_ptr_tpos_proj(obj_pos)
            .unsqueeze(1)
            .expand(
                -1,
                batch_size,
                core.mem_dim,
            )
        )

        # Memory attention works in mem_dim, while object pointers are in
        # hidden_dim.  If necessary, split each pointer into several mem_dim
        # chunks so every chunk becomes a memory token.
        if core.mem_dim < hidden_dim:
            ptr_repeat = hidden_dim // core.mem_dim
            obj_ptrs = obj_ptrs.reshape(-1, batch_size, ptr_repeat, core.mem_dim)
            obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
            obj_pos = (
                obj_pos.unsqueeze(1)
                .expand(-1, ptr_repeat, -1, -1)
                .reshape(-1, batch_size, core.mem_dim)
            )

        return obj_ptrs, obj_pos

    def _condition_on_memory(
        self,
        vision_feats,
        vision_pos_embeds,
        feat_sizes,
    ):
        """Run SAM2 memory attention using the fixed tensorised memory bank."""
        assert self._state is not None
        core = self.model.sam2_model
        batch_size = vision_feats[-1].size(1)
        hidden_dim = core.hidden_dim
        height, width = feat_sizes[-1]

        mask_memory, mask_memory_pos = self._build_mask_memory()
        obj_ptrs, obj_ptr_pos = self._build_object_pointer_memory(batch_size)

        memory = torch.cat([mask_memory, obj_ptrs], dim=0)
        memory_pos = torch.cat([mask_memory_pos, obj_ptr_pos], dim=0)

        attended = core.memory_attention(
            curr=vision_feats[-1],
            curr_pos=vision_pos_embeds[-1],
            memory=memory,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=obj_ptrs.shape[0],
        )
        return attended.permute(1, 2, 0).view(batch_size, hidden_dim, height, width)

    # ---------------------------------------------------------------------
    # 3.3. Forward: decoder
    # ---------------------------------------------------------------------

    def _decode_mask(
        self,
        pix_feat: torch.Tensor,
        vision_feats,
        feat_sizes,
        image_norm: torch.Tensor,
    ):
        """Run prompt encoding, SAM2 mask decoding, upsampling and refinement."""
        model = self.model
        core = model.sam2_model
        batch_size, _, height, width = image_norm.shape

        sparse, dense, image_pe = model._prompt_tensors(pix_feat, batch_size)
        high_res_feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *size)
            for feat, size in zip(vision_feats[:-1], feat_sizes[:-1])
        ]

        low_res_masks, iou_preds, obj_ptr, _ = core.sam_mask_decoder(
            image_embeddings=pix_feat,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=model.multimask_output,
            repeat_image=False,
            high_res_features=high_res_feats,
        )

        if model.multimask_output:
            best = iou_preds.argmax(dim=1)
            rows = torch.arange(batch_size, device=image_norm.device)
            mask_logits = low_res_masks[rows, best].unsqueeze(1)
            obj_ptr = obj_ptr.gather(
                1,
                best.view(batch_size, 1, 1).expand(-1, 1, obj_ptr.shape[-1]),
            )
        else:
            mask_logits = low_res_masks

        obj_ptr = obj_ptr.squeeze(1)

        # DIVERGENCE 5: match SAM-VTP's eager path by resizing to the actual
        # input resolution and using bicubic interpolation at scale >= 3.
        scale = max(height, width) / max(mask_logits.shape[-2], mask_logits.shape[-1])
        mask_logits = F.interpolate(
            mask_logits,
            size=(height, width),
            mode="bicubic" if scale >= 3 else "bilinear",
            align_corners=False,
        )

        if model.refine_head is not None and model.use_refine_head:
            mask_logits = model.refine_head(mask_logits, image_norm)

        return mask_logits, obj_ptr

    # ---------------------------------------------------------------------
    # 3.4. Forward: memory write
    # ---------------------------------------------------------------------

    def _update_memory(
        self,
        vision_feats,
        feat_sizes,
        mask_logits: torch.Tensor,
        obj_ptr: torch.Tensor,
    ) -> None:
        """Encode the current prediction and shift the fixed memory bank."""
        assert self._state is not None
        core = self.model.sam2_model
        state = self._state

        # DIVERGENCE 2: the eager path derives the object score from the
        # predicted mask rather than the decoder's object-score head.
        object_score_logits = torch.any(
            mask_logits.flatten(1) > 0.0,
            dim=1,
            keepdim=True,
        ).float()
        maskmem_features, maskmem_pos_enc = core._encode_new_memory(
            current_vision_feats=vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=mask_logits,
            object_score_logits=object_score_logits,
            is_mask_from_pts=False,
        )

        if self._k > 0:
            # The bank is ordered oldest -> newest.  Drop the oldest entry,
            # shift everything left, then append the current frame.
            state.non_cond_maskmem_features[:-1].copy_(
                state.non_cond_maskmem_features[1:].clone()
            )
            state.non_cond_maskmem_pos_enc[:-1].copy_(
                state.non_cond_maskmem_pos_enc[1:].clone()
            )
            state.non_cond_obj_ptr[:-1].copy_(state.non_cond_obj_ptr[1:].clone())
            state.non_cond_maskmem_features[-1].copy_(maskmem_features)
            state.non_cond_maskmem_pos_enc[-1].copy_(maskmem_pos_enc[-1])
            state.non_cond_obj_ptr[-1].copy_(obj_ptr)

        state.step_idx.add_(1.0)

    # ---------------------------------------------------------------------
    # 3.5. Forward: complete graph-compatible pass
    # ---------------------------------------------------------------------

    def _forward_tensorized_state(
        self,
        frame: torch.Tensor,
        update_memory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run one complete frame using only graph-compatible state."""
        model = self.model
        batch_size = frame.shape[0]

        image_norm, vision_feats, vision_pos_embeds, feat_sizes = self._encode_backbone(
            frame,
            self._static_depth,
            self._static_valid,
        )

        if self._framewise:
            # DIVERGENCE 1: no_mem_embed is written back into vision_feats, as
            # in SAMVTP.forward_frame. Keeping this exact matters: the two
            # implementations have to agree frame for frame.
            vision_feats[-1] = vision_feats[-1] + model.sam2_model.no_mem_embed
            pix_feat = (
                vision_feats[-1].permute(1, 2, 0).view(batch_size, -1, *feat_sizes[-1])
            )
        else:
            pix_feat = self._condition_on_memory(
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
            )

        mask_logits, obj_ptr = self._decode_mask(
            pix_feat,
            vision_feats,
            feat_sizes,
            image_norm,
        )

        if not update_memory:
            # Match the eager path: a non-writing frame stops before memory
            # encoding, bank mutation and frame_idx advancement.
            return {"logits": mask_logits}

        self._update_memory(
            vision_feats,
            feat_sizes,
            mask_logits,
            obj_ptr,
        )
        return {"logits": mask_logits}

    # ---------------------------------------------------------------------
    # 4. Capture
    # ---------------------------------------------------------------------

    def _allocate_static_buffers(
        self,
        frame: torch.Tensor,
        depth: Optional[torch.Tensor],
        depth_valid: Optional[torch.Tensor],
    ) -> None:
        """Allocate the fixed input/output storage used by graph replay."""
        self._static_input = torch.empty_like(frame)
        self._static_depth = torch.empty_like(depth) if depth is not None else None
        self._static_valid = (
            torch.empty_like(depth_valid, dtype=torch.float32)
            if depth_valid is not None
            else None
        )

    def _snapshot_memory_state(self):
        """Save mutable non-conditioning state before graph capture mutates it."""
        if self._state is None:
            return None
        state = self._state
        return (
            state.non_cond_maskmem_features.clone(),
            state.non_cond_maskmem_pos_enc.clone(),
            state.non_cond_obj_ptr.clone(),
            state.step_idx.clone(),
        )

    def _restore_memory_state(self, snapshot) -> None:
        """Restore recurrent state to exactly where capture started."""
        if snapshot is None or self._state is None:
            return
        state = self._state
        state.non_cond_maskmem_features.copy_(snapshot[0])
        state.non_cond_maskmem_pos_enc.copy_(snapshot[1])
        state.non_cond_obj_ptr.copy_(snapshot[2])
        state.step_idx.copy_(snapshot[3])

    def _capture_write_graph(self, capture_stream, writes: bool) -> None:
        """Capture the primary graph on the supplied CUDA stream."""
        self._graph = torch.cuda.CUDAGraph()
        with self._graph_context(self._graph):
            self._static_output = self._forward_tensorized_state(
                self._static_input,
                update_memory=writes,
            )

    def _capture_nowrite_graph(self) -> None:
        """Capture the optional graph for frames that skip the memory write."""
        self._graph_nowrite = torch.cuda.CUDAGraph()
        with self._graph_context(self._graph_nowrite):
            self._static_output_nowrite = self._forward_tensorized_state(
                self._static_input,
                update_memory=False,
            )

    def _capture(
        self,
        frame: torch.Tensor,
        depth: Optional[torch.Tensor],
        depth_valid: Optional[torch.Tensor],
    ) -> None:
        """Warm lazy CUDA state and capture the required graph(s)."""
        assert self._framewise or self._state is not None

        # The primary graph writes memory in full mode.  Frame-wise mode has no
        # memory state, so its only graph is naturally a non-writing graph.
        writes = not self._framewise
        snapshot = self._snapshot_memory_state()

        self._allocate_static_buffers(frame, depth, depth_valid)
        capture_stream = torch.cuda.Stream(device=self.device)

        torch.cuda.synchronize(self.device)
        with torch.cuda.stream(capture_stream):
            self._copy_inputs(frame, depth, depth_valid)

            # Warmup must happen on the capture stream.  This populates lazy
            # Hiera / sine-PE / RoPE caches before graph capture starts.
            for _ in range(self.capture_warmup):
                self._forward_tensorized_state(
                    self._static_input,
                    update_memory=writes,
                )

            self._capture_write_graph(capture_stream, writes)

            # A frame-wise pass already is the non-writing graph. In full mode,
            # capture a separate graph so its replay cannot share mutable graph
            # buffers with the writing graph.
            if self._capture_nowrite and not self._framewise:
                self._capture_nowrite_graph()

        torch.cuda.current_stream(self.device).wait_stream(capture_stream)
        self._restore_memory_state(snapshot)

        self._captured = True
        self._captured_component_epoch = self.model.component_epoch

    def _copy_inputs(
        self,
        frame: torch.Tensor,
        depth: Optional[torch.Tensor],
        depth_valid: Optional[torch.Tensor],
    ) -> None:
        """Copy current inputs into the fixed buffers owned by the graph."""
        assert self._static_input is not None
        self._static_input.copy_(frame)

        if self._static_depth is not None and depth is not None:
            self._static_depth.copy_(depth)

        if self._static_valid is not None and depth_valid is not None:
            self._static_valid.copy_(depth_valid.float())

    # ---------------------------------------------------------------------
    # 5. Frame cadence
    # ---------------------------------------------------------------------

    def _writes_this_frame(self, update_memory: bool) -> bool:
        """Mirror the eager memory-write cadence and advance its frame counter.

        ``SAMVTP.forward_frame`` decides whether to write using ``_frames_seen``
        and increments that counter immediately afterwards. The graph path must
        do exactly the same thing so the eager-to-replay handover stays in phase.
        """
        model = self.model

        if not model.use_memory:
            update_memory = False

        every = int(getattr(model, "memory_write_every", 1))
        if update_memory and every > 1:
            if (model._frames_seen % every) != 0:
                update_memory = False

        model._frames_seen += 1
        return update_memory

    # ---------------------------------------------------------------------
    # 6. Public execution
    # ---------------------------------------------------------------------

    def _run_eager(
        self,
        frame: torch.Tensor,
        is_init_frame: bool,
        update_memory: bool,
        depth: Optional[torch.Tensor],
        depth_valid: Optional[torch.Tensor],
    ):
        """Run through the canonical eager implementation."""
        return self.model.forward_frame(
            frame,
            is_init_frame=is_init_frame,
            update_memory=update_memory,
            depth=depth,
            depth_valid=depth_valid,
        )

    def _check_component_configuration(self) -> None:
        """Prevent silent replay with switches different from capture time."""
        if (
            self._captured
            and self.model.component_epoch != self._captured_component_epoch
        ):
            raise RuntimeError(
                "component switches changed after the CUDA graph was captured; "
                "the graph still holds the old configuration. Call "
                "set_components() before the first frame, or build a new runner."
            )

    def _warmup_eager_frame(
        self,
        frame: torch.Tensor,
        is_init_frame: bool,
        update_memory: bool,
        depth: Optional[torch.Tensor],
        depth_valid: Optional[torch.Tensor],
    ):
        """Run an eager frame while waiting for graph capture to become possible."""
        out = self._run_eager(
            frame,
            is_init_frame,
            update_memory,
            depth,
            depth_valid,
        )

        if self._framewise:
            # frame_idx does not advance with memory disabled, so use the eager
            # frame counter to determine when lazy caches have had enough warmup.
            self._framewise_ready = self.model._frames_seen >= self.min_history_frames
        elif self.model.frame_idx >= self.min_history_frames:
            self._sync_tensor_state_from_model()

        return out

    def _replay_framewise(
        self,
        frame: torch.Tensor,
        depth: Optional[torch.Tensor],
        depth_valid: Optional[torch.Tensor],
    ):
        """Replay the single feed-forward graph."""
        self._copy_inputs(frame, depth, depth_valid)
        self._graph.replay()
        return self._static_output

    def _replay_memory(
        self,
        frame: torch.Tensor,
        update_memory: bool,
        depth: Optional[torch.Tensor],
        depth_valid: Optional[torch.Tensor],
    ):
        """Replay either the writing or non-writing graph."""
        writes = self._writes_this_frame(update_memory)

        if not writes and self._graph_nowrite is None:
            raise RuntimeError(
                "update_memory=False under CUDA-graph replay, but no non-writing "
                "graph was captured. Build the runner with capture_nowrite=True."
            )

        self._copy_inputs(frame, depth, depth_valid)

        if writes:
            self._graph.replay()
            return self._static_output

        self._graph_nowrite.replay()
        return self._static_output_nowrite

    def run(
        self,
        frame: torch.Tensor,
        is_init_frame: bool,
        update_memory: bool = True,
        depth: Optional[torch.Tensor] = None,
        depth_valid: Optional[torch.Tensor] = None,
    ):
        """Run one frame, using eager execution until the graph is ready."""
        self._check_component_configuration()

        if not self.enabled or self._capture_failed:
            return self._run_eager(
                frame,
                is_init_frame,
                update_memory,
                depth,
                depth_valid,
            )

        # The initial frame and all warmup frames use the canonical eager path.
        # This builds SAM2's dictionary memory before we freeze it into tensors.
        if is_init_frame or not self._capture_ready():
            return self._warmup_eager_frame(
                frame,
                is_init_frame,
                update_memory,
                depth,
                depth_valid,
            )

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
                return self._run_eager(
                    frame,
                    is_init_frame,
                    update_memory,
                    depth,
                    depth_valid,
                )

        if self._framewise:
            # Keep _frames_seen aligned with eager execution even though the
            # graph itself has no recurrent memory update.
            self.model._frames_seen += 1
            return self._replay_framewise(frame, depth, depth_valid)

        return self._replay_memory(
            frame,
            update_memory,
            depth,
            depth_valid,
        )
