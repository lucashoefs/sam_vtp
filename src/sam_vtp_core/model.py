from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class SAMVTP(nn.Module):
    """SAM-VTP model that reuses SAM2 memory components for temporal segmentation."""

    def __init__(
        self,
        sam2_cfg_path: str,
        sam2_checkpoint_path: Optional[str] = None,
        freeze_encoder: bool = True,
        freeze_decoder: bool = False,
        num_maskmem: int = 7,
        score_thresh: float = 0.0,
        multimask: bool = False,
    ):
        super().__init__()

        self.num_maskmem = int(num_maskmem)
        self.score_thresh = float(score_thresh)
        self.multimask_output = bool(multimask)
        self.freeze_encoder = bool(freeze_encoder)
        self.input_normalization_mode = "auto"

        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.sam2_model = self._load_sam2_model(
            sam2_cfg_path,
            sam2_checkpoint_path,
            freeze_encoder=freeze_encoder,
            freeze_decoder=freeze_decoder,
        )

        self._prompt_cache: Dict[Any, Dict[str, torch.Tensor]] = {}
        self._image_pe_cache: Dict[Any, torch.Tensor] = {}
        self.prompt_encoder_trainable = any(
            p.requires_grad for p in self.sam2_model.sam_prompt_encoder.parameters()
        )
        self._max_non_cond_frames = max(0, int(self.num_maskmem) - 1)

        self.reset_memory()

    @staticmethod
    def _torch_load(path: str, map_location: str = "cpu") -> Dict[str, Any]:
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=map_location)

    def _load_sam2_model(
        self,
        cfg_path: str,
        checkpoint_path: Optional[str],
        freeze_encoder: bool,
        freeze_decoder: bool,
    ):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate

        GlobalHydra.instance().clear()
        config_dir = str(Path(cfg_path).parent.resolve())
        config_name = Path(cfg_path).stem

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=config_name)
            sam2_model = instantiate(cfg.model, _recursive_=True)

        if checkpoint_path:
            checkpoint = self._torch_load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint)
            model_state = sam2_model.state_dict()
            filtered_state = {
                key: value
                for key, value in state_dict.items()
                if key in model_state and value.shape == model_state[key].shape
            }
            sam2_model.load_state_dict(filtered_state, strict=False)

        if freeze_encoder:
            for name in ["image_encoder", "trunk"]:
                if hasattr(sam2_model, name):
                    for param in getattr(sam2_model, name).parameters():
                        param.requires_grad = False

        if freeze_decoder:
            for name in ["sam_mask_decoder", "sam_prompt_encoder"]:
                if hasattr(sam2_model, name):
                    for param in getattr(sam2_model, name).parameters():
                        param.requires_grad = False

        return sam2_model

    def load_finetuned_checkpoint(
        self,
        checkpoint_path: str,
        strict: bool = False,
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        """Load a SAM-VTP fine-tuned checkpoint into this model instance."""
        checkpoint = self._torch_load(checkpoint_path, map_location=map_location)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
        incompatible = self.load_state_dict(state_dict, strict=strict)
        return {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }

    def reset_memory(self) -> None:
        self.frame_idx = 0
        self.output_dict = {
            "cond_frame_outputs": OrderedDict(),
            "non_cond_frame_outputs": OrderedDict(),
        }

    @staticmethod
    def _device_cache_key(device: torch.device) -> str:
        if device.type != "cuda":
            return device.type
        idx = 0 if device.index is None else int(device.index)
        return "cuda:{}".format(idx)

    @staticmethod
    def _autograd_context(allow_grad: bool):
        if allow_grad and torch.is_grad_enabled():
            return torch.enable_grad()
        return torch.no_grad()

    def _get_bottom_prompts(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        use_cache: bool,
    ):
        if not use_cache:
            point_coords = torch.tensor(
                [
                    [
                        [0.0, float(height - 1)],
                        [float(width - 1), float(height - 1)],
                        [float((width - 1) // 2), float(height - 1)],
                    ]
                ],
                device=device,
                dtype=torch.float32,
            ).repeat(batch_size, 1, 1)
            point_coords = point_coords * (
                float(self.sam2_model.image_size) / float(max(height, width))
            )
            point_labels = torch.ones((batch_size, 3), dtype=torch.int, device=device)
            return point_coords, point_labels

        key = (self._device_cache_key(device), int(height), int(width))
        cached = self._prompt_cache.get(key)
        if cached is None:
            base_coords = torch.tensor(
                [
                    [
                        [0.0, float(height - 1)],
                        [float(width - 1), float(height - 1)],
                        [float((width - 1) // 2), float(height - 1)],
                    ]
                ],
                device=device,
                dtype=torch.float32,
            )
            base_coords = base_coords * (
                float(self.sam2_model.image_size) / float(max(height, width))
            )
            base_labels = torch.ones((1, 3), dtype=torch.int, device=device)
            cached = {"coords": base_coords, "labels": base_labels}
            self._prompt_cache[key] = cached

        return cached["coords"].expand(batch_size, -1, -1), cached["labels"].expand(
            batch_size, -1
        )

    def _get_image_pe(
        self,
        out_h: int,
        out_w: int,
        dtype: torch.dtype,
        device: torch.device,
        use_cache: bool,
    ) -> torch.Tensor:
        if not use_cache:
            image_pe = torch.nn.functional.interpolate(
                self.sam2_model.sam_prompt_encoder.get_dense_pe(),
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            )
            if image_pe.device != device or image_pe.dtype != dtype:
                image_pe = image_pe.to(device=device, dtype=dtype)
            return image_pe

        key = (self._device_cache_key(device), int(out_h), int(out_w), str(dtype))
        cached = self._image_pe_cache.get(key)
        if cached is None:
            image_pe = torch.nn.functional.interpolate(
                self.sam2_model.sam_prompt_encoder.get_dense_pe(),
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            )
            cached = image_pe.to(device=device, dtype=dtype)
            self._image_pe_cache[key] = cached
        return cached

    def _prepare_backbone_features(self, backbone_out):
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.sam2_model.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.sam2_model.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.sam2_model.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def normalize_imagenet(self, image: torch.Tensor) -> torch.Tensor:
        mode = getattr(self, "input_normalization_mode", "auto")
        if mode == "0_255":
            image = image / 255.0
        elif mode == "0_1":
            pass
        elif mode == "-1_1":
            image = (image + 1.0) / 2.0
        else:
            if image.max() > 1.0:
                image = image / 255.0
            elif image.min() < 0.0:
                image = (image + 1.0) / 2.0

        image = image.clamp(0.0, 1.0)
        image = (image - self.pixel_mean) / self.pixel_std
        return image

    def forward_frame(
        self,
        image: torch.Tensor,
        is_init_frame: bool = False,
        use_rolling_memory: bool = True,
        update_memory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        B, _, H, W = image.shape
        image = self.normalize_imagenet(image)

        with self._autograd_context((not self.freeze_encoder) and self.training):
            backbone_out = self.sam2_model.forward_image(image)
            _, vision_feats, vision_pos_embeds, feat_sizes = self._prepare_backbone_features(
                backbone_out
            )

        has_memory = (
            len(self.output_dict["cond_frame_outputs"]) > 0
            or len(self.output_dict["non_cond_frame_outputs"]) > 0
        )

        if (not has_memory) or is_init_frame or self.frame_idx == 0:
            if self.sam2_model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
            pix_feat = vision_feats[-1].permute(1, 2, 0).view(B, -1, *feat_sizes[-1])
        else:
            pix_feat = self.sam2_model._prepare_memory_conditioned_features(
                frame_idx=self.frame_idx,
                is_init_cond_frame=False,
                current_vision_feats=[vision_feats[-1]],
                current_vision_pos_embeds=[vision_pos_embeds[-1]],
                feat_sizes=[feat_sizes[-1]],
                output_dict=self.output_dict,
                num_frames=self.frame_idx + 1,
                track_in_reverse=False,
            )

        use_static_cache = (not self.training) and (not torch.is_grad_enabled())
        point_coords, point_labels = self._get_bottom_prompts(
            B,
            H,
            W,
            image.device,
            use_cache=use_static_cache,
        )

        grad_context = self._autograd_context(self.prompt_encoder_trainable and self.training)
        with grad_context:
            sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )
            if dense_embeddings.shape[-2:] != pix_feat.shape[-2:]:
                dense_embeddings = torch.nn.functional.interpolate(
                    dense_embeddings,
                    size=pix_feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

        high_res_feats = [
            feat.permute(1, 2, 0).view(B, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[:-1], feat_sizes[:-1])
        ]

        image_pe = self._get_image_pe(
            out_h=int(pix_feat.shape[-2]),
            out_w=int(pix_feat.shape[-1]),
            dtype=pix_feat.dtype,
            device=pix_feat.device,
            use_cache=use_static_cache,
        )

        low_res_masks, iou_preds, obj_ptr, _ = self.sam2_model.sam_mask_decoder(
            image_embeddings=pix_feat,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
            repeat_image=False,
            high_res_features=high_res_feats,
        )

        if self.multimask_output:
            best_idx = iou_preds.argmax(dim=1)
            batch_indices = torch.arange(B, device=image.device)
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
        mask_logits = torch.nn.functional.interpolate(
            mask_logits,
            size=(H, W),
            mode=interpolation_mode,
            align_corners=False,
        )

        if not update_memory:
            return {"logits": mask_logits}

        is_obj_appearing = torch.any(mask_logits.flatten(1) > 0.0, dim=1, keepdim=True)
        object_score_logits = is_obj_appearing.float()

        maskmem_features, maskmem_pos_enc = self.sam2_model._encode_new_memory(
            current_vision_feats=vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=mask_logits,
            object_score_logits=object_score_logits,
            is_mask_from_pts=True,
        )

        memory_dict = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "obj_ptr": obj_ptr,
        }

        if use_rolling_memory:
            if self.frame_idx == 0:
                self.output_dict["cond_frame_outputs"][self.frame_idx] = memory_dict
            else:
                self.output_dict["non_cond_frame_outputs"][self.frame_idx] = memory_dict
                if (
                    len(self.output_dict["non_cond_frame_outputs"])
                    > self._max_non_cond_frames
                ):
                    self.output_dict["non_cond_frame_outputs"].popitem(last=False)
        else:
            if is_init_frame:
                self.output_dict["cond_frame_outputs"][self.frame_idx] = memory_dict
            else:
                self.output_dict["non_cond_frame_outputs"][self.frame_idx] = memory_dict
                if (
                    len(self.output_dict["non_cond_frame_outputs"])
                    > self._max_non_cond_frames
                ):
                    self.output_dict["non_cond_frame_outputs"].popitem(last=False)

        self.frame_idx += 1
        return {"logits": mask_logits}

    def forward(
        self,
        image_sequence: torch.Tensor,
        reset_memory: bool = True,
        use_rolling_memory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        _, T, _, _, _ = image_sequence.shape

        if reset_memory:
            self.reset_memory()

        all_logits = []
        for t in range(T):
            frame = image_sequence[:, t]
            is_init = self.frame_idx == 0
            output = self.forward_frame(
                frame,
                is_init_frame=is_init,
                use_rolling_memory=use_rolling_memory,
            )
            all_logits.append(output["logits"])

        return {"logits": torch.stack(all_logits, dim=1)}

    def predict_tensor(
        self,
        image: torch.Tensor,
        use_rolling_memory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        return self.forward_frame(
            image,
            is_init_frame=(self.frame_idx == 0),
            use_rolling_memory=use_rolling_memory,
        )
