from __future__ import annotations
 
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from sam_vtp.modeling.geometry import TARTANGROUND, CameraIntrinsics, geometry_channels
 
 
class SAMVTP(nn.Module):
 
    COMPONENTS: Tuple[str, ...] = ("memory", "geometry", "refine")
 
    def __init__(
        self,
        sam2_model,
        geometry_encoder=None,
        refine_head=None,
        camera: Optional[CameraIntrinsics] = None,
        geometry_hw: int = 256,
        memory_write_every: int = 1,
        input_normalization_mode: str = "0_255",
    ):
        super().__init__()
        self.sam2_model = sam2_model
        self.geometry_encoder = geometry_encoder
        self.refine_head = refine_head
 
        # ImageNet statistics in 0-1 space. normalize_imagenet() brings the
        # input into that range first, so the 0-255 division happens upstream
        # of these. Persistent by default, and both are in the manifest --
        # persistent=False would give you two missing keys.
        self.register_buffer("pixel_mean",
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std",
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
 
        # --- switches -------------------------------------------------------
        # Two distinct notions that must not share a variable. `_trained_*` is
        # what the checkpoint carries, fixed at construction. `use_*` is what is
        # running right now. An ablation flips the second on a fully-loaded
        # model; it never builds a smaller one.
        self.use_memory = sam2_model.memory_attention is not None
        self.use_geometry = geometry_encoder is not None
        self.use_refine_head = refine_head is not None
        self._trained_components = {
            "memory": self.use_memory,
            "geometry": self.use_geometry,
            "refine": self.use_refine_head,
        }
        self._component_epoch = 0
 
        # --- frame-loop configuration --------------------------------------
        self.num_maskmem = sam2_model.num_maskmem
        self._max_non_cond_frames = max(0, self.num_maskmem - 1)
        self.multimask_output = bool(sam2_model.multimask_output_in_sam)
        self.memory_write_every = int(memory_write_every)
        self.input_normalization_mode = input_normalization_mode
        self.geometry_hw = int(geometry_hw)
        self.camera = camera if camera is not None else TARTANGROUND
 
        # --- runtime state (never in state_dict) ----------------------------
        # These exact names are the contract fullgraph.py reads.
        self.output_dict: Dict[str, "OrderedDict[int, Dict[str, Any]]"] = {
            "cond_frame_outputs": OrderedDict(),
            "non_cond_frame_outputs": OrderedDict(),
        }
        self.frame_idx = 0
        self._frames_seen = 0
        self._prompt_cache: Dict[Any, Any] = {}
 
    @property
    def component_epoch(self) -> int:
        """Bumped on every switch change, so a captured graph can detect staleness."""
        return self._component_epoch
 
    @property
    def trained_components(self) -> Dict[str, bool]:
        return dict(self._trained_components)
 
    @property
    def active_components(self) -> Dict[str, bool]:
        return {"memory": bool(self.use_memory),
                "geometry": bool(self.use_geometry),
                "refine": bool(self.use_refine_head and self.refine_head is not None)}
 
    @property
    def component_deviations(self) -> Dict[str, bool]:
        """Non-empty means these numbers are an ablation, not the model as trained."""
        return {k: v for k, v in self.active_components.items()
                if self._trained_components.get(k) != v}
 
    @property
    def needs_depth(self) -> bool:
        return bool(self.use_geometry and self.geometry_encoder is not None)

    @property
    def needs_pose(self) -> bool:
        """Always False. Retained so existing callers keep working.

        Nothing in this model reads a camera pose. The ego-motion warp that once
        did was removed; it cost a localisation dependency and did not measurably
        improve accuracy. Callers may stop looking up transforms entirely.
        """
        return False

    @property
    def needs_geometry(self) -> bool:
        """Deprecated alias for `needs_depth`.

        Kept because callers outside this repo test it before supplying inputs.
        """
        return self.needs_depth

    @property
    def memory_fill_level(self) -> float:
        """Fraction of the memory bank that holds real frames, in [0, 1].

        Predictions before the bank is full are made from less temporal context
        than the model was trained with. A consumer can gate planning on this
        rather than trusting the first frames of a stream.
        """
        n = (len(self.output_dict["cond_frame_outputs"])
             + len(self.output_dict["non_cond_frame_outputs"]))
        return min(1.0, n / max(1, self.num_maskmem))
 
    def set_components(self, *, memory=None, geometry=None, refine=None) -> "SAMVTP":
        """Flip switches at runtime. None leaves a component unchanged."""
        requested = {k: v for k, v in
                     (("memory", memory), ("geometry", geometry), ("refine", refine))
                     if v is not None}
        if not requested:
            return self
 
        absent = {"memory": self.sam2_model.memory_attention is None,
                  "geometry": self.geometry_encoder is None,
                  "refine": self.refine_head is None}
        for name, want in requested.items():
            if want and absent[name]:
                raise ValueError(
                    f"cannot enable {name!r}: this checkpoint carries no weights "
                    f"for it (trained with {self._trained_components})")
 
        before = self.active_components
        if "memory" in requested:
            self.use_memory = bool(requested["memory"])
        if "geometry" in requested:
            self.use_geometry = bool(requested["geometry"])
        if "refine" in requested:
            self.use_refine_head = bool(requested["refine"])
 
        if self.active_components != before:
            self._component_epoch += 1
        # Only the memory switch invalidates the bank. Geometry and refine can
        # be toggled mid-stream without losing history: neither participates in
        # what the bank stores.
        if before["memory"] != self.use_memory:
            self.reset_memory()
        return self

    @contextmanager
    def components(self, **switches: Optional[bool]) -> Iterator["SAMVTP"]:
        """Apply switches for a block, then restore them.

            with model.components(geometry=False):
                rgb_only = model.forward_frame(rgb, is_init_frame=True)

        Memory is reset on entry and on exit whenever the temporal path changes,
        so the block cannot inherit or leak a bank written under the other
        configuration.
        """
        previous = self.active_components
        try:
            yield self.set_components(**switches)
        finally:
            self.set_components(**previous)
 
    def reset_memory(self) -> None:
        self.output_dict["cond_frame_outputs"].clear()
        self.output_dict["non_cond_frame_outputs"].clear()
        self.frame_idx = 0
        self._frames_seen = 0
 
    def clear_inference_caches(self) -> None:
        """Drop every cached tensor derived from weights."""
        self.sam2_model.image_encoder.trunk._pos_embed_cache.clear()
        self._prompt_cache.clear()
 
    @torch.no_grad()
    def warmup(self, n: int = 3, hw: Tuple[int, int] = (512, 512)) -> None:
        """Run synthetic frames to pay lazy initialisation cost off the clock.

        The first inference allocates cuDNN workspaces, chooses convolution
        algorithms and fills the position-encoding and rotary caches, which
        together cost far more than a steady-state frame. Call this after
        loading and before the first real frame, or the robot pays it live.

        Resets memory afterwards, so the bank does not start full of noise.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        image = torch.zeros(1, 3, int(hw[0]), int(hw[1]), device=device, dtype=dtype)
        depth = None
        if self.needs_depth:
            depth = torch.full((1, 1, int(hw[0]), int(hw[1])), 5.0,
                               device=device, dtype=dtype)
        was_training = self.training
        self.eval()
        for i in range(max(1, int(n))):
            self.forward_frame(image, is_init_frame=(i == 0), depth=depth)
        self.reset_memory()
        if was_training:
            self.train()

    def describe(self) -> str:
        """One-line summary, for logs that need to record what actually ran.

        Ends with an explicit ABLATED clause whenever a component is switched
        away from how the weights were trained, so a log line shows, rather than
        assumes, that a number came from the full model.
        """
        n = sum(p.numel() for p in self.parameters()) / 1e6
        active = self.active_components
        bits = [f"SAM-VTP {n:.1f}M params",
                "memory " + (f"on, {self.num_maskmem} frames" if active["memory"]
                             else "OFF (frame-wise)"),
                f"geometry {'on' if active['geometry'] else 'off'}",
                f"refine head {'on' if active['refine'] else 'off'}",
                f"multimask {'on' if self.multimask_output else 'off'}"]
        if self.memory_write_every > 1:
            bits.append(f"memory write every {self.memory_write_every}")
        out = ", ".join(bits)
        dev = self.component_deviations
        if dev:
            out += "  [ABLATED: " + ", ".join(
                f"{k} {'on' if v else 'off'} (trained "
                f"{'on' if self._trained_components[k] else 'off'})"
                for k, v in sorted(dev.items())) + "]"
        return out

    def normalize_imagenet(self, image: torch.Tensor) -> torch.Tensor:
        mode = self.input_normalization_mode
        if mode == "0_255":
            image = image / 255.0
        elif mode == "0_1":
            pass
        elif mode == "-1_1":
            image = (image + 1.0) / 2.0
        elif mode == "auto":
            if image.max() > 1.0:
                image = image / 255.0
            elif image.min() < 0.0:
                image = (image + 1.0) / 2.0
        else:
            raise ValueError(f"unknown input_normalization_mode {mode!r}")
        return (image.clamp(0.0, 1.0) - self.pixel_mean) / self.pixel_std
 
    def _prompt_tensors(self, pix_feat: torch.Tensor, B: int):
        """DIVERGENCE 4: resample the learned prompt to the running feature grid."""
        size = tuple(pix_feat.shape[-2:])
 
        def build():
            enc = self.sam2_model.sam_prompt_encoder
            sparse, dense = enc(points=None, boxes=None, masks=None)
            if dense.shape[-2:] != size:
                dense = F.interpolate(dense, size=size, mode="bilinear",
                                      align_corners=False)
            pe = enc.get_dense_pe()
            if pe.shape[-2:] != size:
                pe = F.interpolate(pe, size=size, mode="bilinear",
                                   align_corners=False)
            return sparse, dense, pe.to(device=pix_feat.device, dtype=pix_feat.dtype)
 
        if self.training or torch.is_grad_enabled():
            sparse, dense, pe = build()
        else:
            # Keyed on the parameter's device and dtype too: under autocast
            # those differ from the feature's, and a .half() on the model would
            # otherwise leave a stale fp32 entry reachable.
            param = self.sam2_model.sam_prompt_encoder.dense_emb
            key = (size, str(pix_feat.device), pix_feat.dtype,
                   str(param.device), param.dtype)
            cached = self._prompt_cache.get(key)
            if cached is None:
                cached = build()
                self._prompt_cache[key] = cached
            sparse, dense, pe = cached
 
        return sparse.expand(B, -1, -1), dense.expand(B, -1, -1, -1), pe
 
    def forward_frame(
        self,
        image: torch.Tensor,
        is_init_frame: bool = False,
        update_memory: bool = True,
        depth: Optional[torch.Tensor] = None,
        depth_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.needs_depth and depth is None:
            raise ValueError(
                "this checkpoint was trained with geometry, so it needs depth in "
                "METRES per frame. Pass it, or call set_components(geometry=False) "
                "to run the RGB-only ablation deliberately.")
 
        B, _, H, W = image.shape
        image_norm = self.normalize_imagenet(image)
 
        # 1. encoder. Inherited unchanged, it already applies the decoder's
        #    conv_s0 / conv_s1 to backbone_fpn[0] and [1].
        backbone_out = self.sam2_model.forward_image(image_norm)
 
        # 2. geometry, into the coarsest level only. That level becomes
        #    vision_feats[-1], so a single insertion point reaches both the
        #    memory conditioning and the decoder.
        if self.use_geometry and depth is not None:
            top = backbone_out["backbone_fpn"][-1]
            geo = geometry_channels(depth,
                                    out_hw=(self.geometry_hw, self.geometry_hw),
                                    cam=self.camera, depth_valid=depth_valid)
            backbone_out["backbone_fpn"] = list(backbone_out["backbone_fpn"])
            backbone_out["backbone_fpn"][-1] = top + self.geometry_encoder(
                geo.to(top.dtype), top.shape[-2:])
 
        # 3. DIVERGENCE 6: SAM 2 returns FOUR values here. Discard the first.
        _, vision_feats, vision_pos_embeds, feat_sizes = \
            self.sam2_model._prepare_backbone_features(backbone_out)
 
        has_memory = bool(self.output_dict["cond_frame_outputs"]
                          or self.output_dict["non_cond_frame_outputs"])
 
        if (not self.use_memory) or (not has_memory) or is_init_frame or self.frame_idx == 0:
            # DIVERGENCE 1: no_mem_embed is written back into vision_feats, not
            # only into pix_feat. Upstream keeps the two separate, so its stored
            # memory omits this term. The conditioning frame's memory depends on
            # it, which moves every frame from 1 onward.
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
            pix_feat = vision_feats[-1].permute(1, 2, 0).view(B, -1, *feat_sizes[-1])
        else:
            # Divergences 3 and 4 live inside this call. See sam_vtp_base.py
            pix_feat = self.sam2_model.prepare_memory_conditioned_features(
                frame_idx=self.frame_idx,
                current_vision_feats=[vision_feats[-1]],
                current_vision_pos_embeds=[vision_pos_embeds[-1]],
                feat_sizes=[feat_sizes[-1]],
                output_dict=self.output_dict,
            )
 
        # 4. decoder, called directly. Upstream's _forward_sam_heads is not used:
        #    it would redo the prompt encoding and upsample to self.image_size.
        sparse_emb, dense_emb, image_pe = self._prompt_tensors(pix_feat, B)
        high_res_feats = [f.permute(1, 2, 0).view(B, -1, *s)
                          for f, s in zip(vision_feats[:-1], feat_sizes[:-1])]
 
        low_res_masks, iou_preds, obj_ptr, _ = self.sam2_model.sam_mask_decoder(
            image_embeddings=pix_feat,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=self.multimask_output,
            repeat_image=False,
            high_res_features=high_res_feats,
        )
 
        if self.multimask_output:
            best = iou_preds.argmax(dim=1)
            rows = torch.arange(B, device=image.device)
            mask_logits = low_res_masks[rows, best].unsqueeze(1)
            obj_ptr = obj_ptr.gather(
                1, best.view(B, 1, 1).expand(-1, 1, obj_ptr.shape[-1]))
        else:
            mask_logits = low_res_masks
        obj_ptr = obj_ptr.squeeze(1)
 
        # 5. DIVERGENCE 5: upsample to the INPUT size, not self.image_size, and
        #    switch to bicubic at scale >= 3. At 512 from 128 the scale is 4, so
        #    this is the bicubic branch.
        scale = max(H, W) / max(mask_logits.shape[-2], mask_logits.shape[-1])
        mask_logits = F.interpolate(
            mask_logits, size=(H, W),
            mode="bicubic" if scale >= 3 else "bilinear", align_corners=False)
 
        if self.refine_head is not None and self.use_refine_head:
            mask_logits = self.refine_head(mask_logits, image_norm)
 
        # 6. memory write
        if not self.use_memory:
            update_memory = False
        if update_memory and self.memory_write_every > 1:
            if (self._frames_seen % self.memory_write_every) != 0:
                update_memory = False
        self._frames_seen += 1
 
        if not update_memory:
            return {"logits": mask_logits}
 
        # DIVERGENCE 2: object_score_logits comes from the prediction itself,
        # not from the decoder's pred_obj_score_head. _encode_new_memory is
        # inherited unchanged -- the divergence is entirely in this caller.
        object_score_logits = torch.any(
            mask_logits.flatten(1) > 0.0, dim=1, keepdim=True).float()
        maskmem_features, maskmem_pos_enc = self.sam2_model._encode_new_memory(
            current_vision_feats=vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=mask_logits,
            object_score_logits=object_score_logits,
            is_mask_from_pts=False,
        )
 
        entry = {"maskmem_features": maskmem_features,
                 "maskmem_pos_enc": maskmem_pos_enc,
                 "obj_ptr": obj_ptr}
        if self.frame_idx == 0:
            self.output_dict["cond_frame_outputs"][self.frame_idx] = entry
        else:
            self.output_dict["non_cond_frame_outputs"][self.frame_idx] = entry
            if len(self.output_dict["non_cond_frame_outputs"]) > self._max_non_cond_frames:
                self.output_dict["non_cond_frame_outputs"].popitem(last=False)
 
        self.frame_idx += 1
        return {"logits": mask_logits}
 
    def forward(self, image_sequence, depth_sequence=None, reset_memory=True):
        if reset_memory:
            self.reset_memory()
        logits = []
        for t in range(image_sequence.shape[1]):
            out = self.forward_frame(
                image_sequence[:, t], is_init_frame=(t == 0),
                depth=None if depth_sequence is None else depth_sequence[:, t])
            logits.append(out["logits"])
        return {"logits": torch.stack(logits, dim=1)}