from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from sam2.modeling.sam2_utils import get_1d_sine_pe

from .model import SAMVTP


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
    """Full-forward CUDA graph replay for SAM-VTP with tensorized recurrent state."""

    def __init__(
        self,
        model: SAMVTP,
        device: torch.device,
        enabled: bool,
        capture_warmup: int = 0,
        min_history_frames: Optional[int] = None,
    ):
        self.model = model
        self.device = device
        self.enabled = bool(enabled) and (device.type == "cuda") and hasattr(torch.cuda, "CUDAGraph")
        self.capture_warmup = max(0, int(capture_warmup))

        default_min_history = max(int(self.model.num_maskmem), 16)
        self.min_history_frames = int(min_history_frames) if min_history_frames is not None else default_min_history
        self.min_history_frames = max(default_min_history, self.min_history_frames)

        self._captured = False
        self._capture_failed = False
        self._capture_error = ""
        self._graph = None

        self._state: Optional[_TensorState] = None
        self._static_input: Optional[torch.Tensor] = None
        self._static_output: Optional[Dict[str, torch.Tensor]] = None

        self._k = max(0, int(self.model.num_maskmem) - 1)
        self._obj_ptr_tdiff_max = max(1.0, float(self.model.sam2_model.max_obj_ptrs_in_encoder - 1))
        self._noncond_tdiff = torch.arange(1, self._k + 1, device=self.device, dtype=torch.float32)

    def status_note(self) -> str:
        if not self.enabled:
            return "CUDA Graph disabled"
        if self._capture_failed:
            return "CUDA Graph fallback to eager (capture failed: {})".format(self._capture_error)
        if self._captured:
            return "CUDA Graph replay active for full SAM-VTP forward with in-graph tensorized state updates"
        return "CUDA Graph enabled (history warmup/capture pending)"

    @staticmethod
    def _graph_context(graph: torch.cuda.CUDAGraph):
        try:
            return torch.cuda.graph(graph, capture_error_mode="thread_local")
        except TypeError:
            return torch.cuda.graph(graph)

    def _sync_tensor_state_from_model(self) -> bool:
        cond = self.model.output_dict.get("cond_frame_outputs", {})
        non_cond = self.model.output_dict.get("non_cond_frame_outputs", {})
        if (0 not in cond) or (self._k > 0 and len(non_cond) < self._k):
            return False

        cond0 = cond[0]
        cond_maskmem_features = cond0["maskmem_features"].to(self.device)
        cond_maskmem_pos_enc = cond0["maskmem_pos_enc"][-1].to(self.device)
        cond_obj_ptr = cond0["obj_ptr"].to(self.device)

        if self._k > 0:
            non_cond_items = sorted(non_cond.items(), key=lambda kv: kv[0])[-self._k :]
            nc_feats = torch.stack([it[1]["maskmem_features"].to(self.device) for it in non_cond_items], dim=0)
            nc_pos = torch.stack([it[1]["maskmem_pos_enc"][-1].to(self.device) for it in non_cond_items], dim=0)
            nc_ptr = torch.stack([it[1]["obj_ptr"].to(self.device) for it in non_cond_items], dim=0)
        else:
            bsz, cm, hm, wm = cond_maskmem_features.shape
            cptr = cond_obj_ptr.shape[-1]
            nc_feats = cond_maskmem_features.new_zeros((0, bsz, cm, hm, wm))
            nc_pos = cond_maskmem_pos_enc.new_zeros((0, bsz, cm, hm, wm))
            nc_ptr = cond_obj_ptr.new_zeros((0, bsz, cptr))

        step_idx = torch.tensor(float(self.model.frame_idx), device=self.device, dtype=torch.float32)

        self._state = _TensorState(
            cond_maskmem_features=cond_maskmem_features,
            cond_maskmem_pos_enc=cond_maskmem_pos_enc,
            cond_obj_ptr=cond_obj_ptr,
            non_cond_maskmem_features=nc_feats,
            non_cond_maskmem_pos_enc=nc_pos,
            non_cond_obj_ptr=nc_ptr,
            step_idx=step_idx,
        )
        return True

    def _prepare_memory_conditioned_pix_feat(self, vision_feats, vision_pos_embeds, feat_sizes) -> torch.Tensor:
        assert self._state is not None

        B = vision_feats[-1].size(1)
        C = self.model.sam2_model.hidden_dim
        H, W = feat_sizes[-1]

        to_cat_memory = []
        to_cat_memory_pos = []

        cond_feat = self._state.cond_maskmem_features.flatten(2).permute(2, 0, 1)
        cond_pos = self._state.cond_maskmem_pos_enc.flatten(2).permute(2, 0, 1)
        cond_pos = cond_pos + self.model.sam2_model.maskmem_tpos_enc[self.model.num_maskmem - 1]
        to_cat_memory.append(cond_feat)
        to_cat_memory_pos.append(cond_pos)

        for t_pos in range(1, self.model.num_maskmem):
            slot = t_pos - 1
            feat = self._state.non_cond_maskmem_features[slot].flatten(2).permute(2, 0, 1)
            pos = self._state.non_cond_maskmem_pos_enc[slot].flatten(2).permute(2, 0, 1)
            pos = pos + self.model.sam2_model.maskmem_tpos_enc[self.model.num_maskmem - t_pos - 1]
            to_cat_memory.append(feat)
            to_cat_memory_pos.append(pos)

        num_obj_ptr_tokens = 0
        if self.model.sam2_model.use_obj_ptrs_in_encoder:
            obj_ptrs = torch.cat(
                [
                    self._state.cond_obj_ptr.unsqueeze(0),
                    torch.flip(self._state.non_cond_obj_ptr, dims=[0]),
                ],
                dim=0,
            )

            if self.model.sam2_model.add_tpos_enc_to_obj_ptrs:
                cond_tdiff = self._state.step_idx.view(1)
                tdiff = torch.cat([cond_tdiff, self._noncond_tdiff], dim=0)
                tpos_dim = C if self.model.sam2_model.proj_tpos_enc_in_obj_ptrs else self.model.sam2_model.mem_dim
                obj_pos = get_1d_sine_pe(tdiff / self._obj_ptr_tdiff_max, dim=tpos_dim)
                obj_pos = self.model.sam2_model.obj_ptr_tpos_proj(obj_pos)
                obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.model.sam2_model.mem_dim)
            else:
                obj_pos = obj_ptrs.new_zeros(obj_ptrs.shape[0], B, self.model.sam2_model.mem_dim)

            if self.model.sam2_model.mem_dim < C:
                ptr_repeat = C // self.model.sam2_model.mem_dim
                obj_ptrs = obj_ptrs.reshape(-1, B, ptr_repeat, self.model.sam2_model.mem_dim)
                obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                obj_pos = obj_pos.unsqueeze(1).expand(-1, ptr_repeat, -1, -1).reshape(-1, B, self.model.sam2_model.mem_dim)

            to_cat_memory.append(obj_ptrs)
            to_cat_memory_pos.append(obj_pos)
            num_obj_ptr_tokens = obj_ptrs.shape[0]

        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos = torch.cat(to_cat_memory_pos, dim=0)

        pix_feat = self.model.sam2_model.memory_attention(
            curr=[vision_feats[-1]],
            curr_pos=[vision_pos_embeds[-1]],
            memory=memory,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        pix_feat = pix_feat.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat

    def _forward_tensorized_state(self, frame: torch.Tensor, use_rolling_memory: bool) -> Dict[str, torch.Tensor]:
        assert self._state is not None
        if not use_rolling_memory:
            raise RuntimeError("SAMVTPFullGraphReplayRunner currently supports use_rolling_memory=True only.")

        B, _, H, W = frame.shape
        image = self.model.normalize_imagenet(frame)

        with self.model._autograd_context((not self.model.freeze_encoder) and self.model.training):
            backbone_out = self.model.sam2_model.forward_image(image)
            _, vision_feats, vision_pos_embeds, feat_sizes = self.model._prepare_backbone_features(backbone_out)

        pix_feat = self._prepare_memory_conditioned_pix_feat(vision_feats, vision_pos_embeds, feat_sizes)

        use_static_cache = (not self.model.training) and (not torch.is_grad_enabled())
        point_coords, point_labels = self.model._get_bottom_prompts(B, H, W, frame.device, use_cache=use_static_cache)

        grad_context = self.model._autograd_context(self.model.prompt_encoder_trainable and self.model.training)
        with grad_context:
            sparse_embeddings, dense_embeddings = self.model.sam2_model.sam_prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )
            if dense_embeddings.shape[-2:] != pix_feat.shape[-2:]:
                dense_embeddings = F.interpolate(
                    dense_embeddings,
                    size=pix_feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

        high_res_feats = [
            feat.permute(1, 2, 0).view(B, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[:-1], feat_sizes[:-1])
        ]

        image_pe = self.model._get_image_pe(
            out_h=int(pix_feat.shape[-2]),
            out_w=int(pix_feat.shape[-1]),
            dtype=pix_feat.dtype,
            device=pix_feat.device,
            use_cache=use_static_cache,
        )

        low_res_masks, iou_preds, obj_ptr, _ = self.model.sam2_model.sam_mask_decoder(
            image_embeddings=pix_feat,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.model.multimask_output,
            repeat_image=False,
            high_res_features=high_res_feats,
        )

        if self.model.multimask_output:
            best_idx = iou_preds.argmax(dim=1)
            batch_indices = torch.arange(B, device=frame.device)
            mask_logits = low_res_masks[batch_indices, best_idx].unsqueeze(1)
            obj_ptr = obj_ptr.gather(
                1,
                best_idx.view(B, 1, 1).expand(-1, 1, obj_ptr.shape[-1]),
            )
        else:
            mask_logits = low_res_masks

        obj_ptr = obj_ptr.squeeze(1)

        upscale_factor = max(H, W) / max(mask_logits.shape[-2], mask_logits.shape[-1])
        interpolation_mode = "bicubic" if upscale_factor >= 3 else "bilinear"
        mask_logits = F.interpolate(mask_logits, size=(H, W), mode=interpolation_mode, align_corners=False)

        is_obj_appearing = torch.any(mask_logits.flatten(1) > 0.0, dim=1, keepdim=True)
        object_score_logits = is_obj_appearing.float()

        maskmem_features, maskmem_pos_enc = self.model.sam2_model._encode_new_memory(
            current_vision_feats=vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=mask_logits,
            object_score_logits=object_score_logits,
            is_mask_from_pts=True,
        )

        if self._k > 0:
            self._state.non_cond_maskmem_features[:-1].copy_(self._state.non_cond_maskmem_features[1:].clone())
            self._state.non_cond_maskmem_pos_enc[:-1].copy_(self._state.non_cond_maskmem_pos_enc[1:].clone())
            self._state.non_cond_obj_ptr[:-1].copy_(self._state.non_cond_obj_ptr[1:].clone())

            self._state.non_cond_maskmem_features[-1].copy_(maskmem_features)
            self._state.non_cond_maskmem_pos_enc[-1].copy_(maskmem_pos_enc[-1])
            self._state.non_cond_obj_ptr[-1].copy_(obj_ptr)

        self._state.step_idx.add_(1.0)

        return {"logits": mask_logits}

    def _capture(self, frame: torch.Tensor, use_rolling_memory: bool) -> None:
        assert self._state is not None

        snap_non_cond_feats = self._state.non_cond_maskmem_features.clone()
        snap_non_cond_pos = self._state.non_cond_maskmem_pos_enc.clone()
        snap_non_cond_ptr = self._state.non_cond_obj_ptr.clone()
        snap_step_idx = self._state.step_idx.clone()

        self._static_input = torch.empty_like(frame)

        capture_stream = torch.cuda.Stream(device=self.device)
        torch.cuda.synchronize(self.device)

        with torch.cuda.stream(capture_stream):
            self._static_input.copy_(frame)

            if self.capture_warmup > 0:
                for _ in range(self.capture_warmup):
                    _ = self._forward_tensorized_state(
                        self._static_input,
                        use_rolling_memory=use_rolling_memory,
                    )

            self._graph = torch.cuda.CUDAGraph()
            with self._graph_context(self._graph):
                self._static_output = self._forward_tensorized_state(
                    self._static_input,
                    use_rolling_memory=use_rolling_memory,
                )

        torch.cuda.current_stream(self.device).wait_stream(capture_stream)

        self._state.non_cond_maskmem_features.copy_(snap_non_cond_feats)
        self._state.non_cond_maskmem_pos_enc.copy_(snap_non_cond_pos)
        self._state.non_cond_obj_ptr.copy_(snap_non_cond_ptr)
        self._state.step_idx.copy_(snap_step_idx)

        self._captured = True

    def _warmup_eager(self, frame: torch.Tensor, is_init_frame: bool, use_rolling_memory: bool):
        out = self.model.forward_frame(
            frame,
            is_init_frame=is_init_frame,
            use_rolling_memory=use_rolling_memory,
            update_memory=True,
        )
        if self.model.frame_idx >= self.min_history_frames:
            _ = self._sync_tensor_state_from_model()
        return out

    def run(self, frame: torch.Tensor, is_init_frame: bool, use_rolling_memory: bool = True):
        if (not self.enabled) or self._capture_failed:
            return self.model.forward_frame(
                frame,
                is_init_frame=is_init_frame,
                use_rolling_memory=use_rolling_memory,
                update_memory=True,
            )

        if is_init_frame or (self._state is None):
            return self._warmup_eager(
                frame,
                is_init_frame=is_init_frame,
                use_rolling_memory=use_rolling_memory,
            )

        if not self._captured:
            try:
                self._capture(frame, use_rolling_memory=use_rolling_memory)
                self._static_input.copy_(frame)
                self._graph.replay()
                return self._static_output
            except Exception as exc:
                self._capture_failed = True
                self._capture_error = "{}: {}".format(type(exc).__name__, str(exc))
                try:
                    torch.cuda.synchronize(self.device)
                except Exception:
                    pass
                return self.model.forward_frame(
                    frame,
                    is_init_frame=is_init_frame,
                    use_rolling_memory=use_rolling_memory,
                    update_memory=True,
                )

        self._static_input.copy_(frame)
        self._graph.replay()
        return self._static_output
