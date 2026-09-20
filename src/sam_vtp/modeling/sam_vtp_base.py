import math
from typing import Any, Dict, List, Tuple

import torch

from sam2.modeling.sam2_base import SAM2Base
from sam_vtp.modeling.prompt import LearnedPrompt


class SAMVTPBase(SAM2Base):
    def __init__(self, *, num_prompt_tokens=4, **kwargs):
        self._num_prompt_tokens = num_prompt_tokens  # BEFORE super()
        super().__init__(**kwargs)

    def _build_sam_heads(self):
        super()._build_sam_heads()
        g = self.sam_image_embedding_size
        self.sam_prompt_encoder = LearnedPrompt(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(g, g),
            input_image_size=(self.image_size, self.image_size),
            num_tokens=self._num_prompt_tokens,
        )

    def prepare_memory_conditioned_features(
        self,
        frame_idx: int,
        current_vision_feats: List[torch.Tensor],
        current_vision_pos_embeds: List[torch.Tensor],
        feat_sizes: List[Tuple[int, int]],
        output_dict: Dict[str, Any],
    ) -> torch.Tensor:
        """Cross-attend the current frame against the memory bank."""
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device

        to_cat_memory, to_cat_memory_pos = [], []
        cond_outputs = output_dict["cond_frame_outputs"]
        non_cond_outputs = output_dict["non_cond_frame_outputs"]
        assert len(cond_outputs) > 0, "memory conditioning needs a conditioning frame"

        # DIVERGENCE 4 (slot selection): unit stride, written out directly.
        # Upstream walks r = memory_temporal_stride_for_eval; with r == 1 the two
        # agree, which is why that flag is pinned in the YAML.
        t_pos_and_prevs = [(0, out) for out in cond_outputs.values()]
        for t_pos in range(1, self.num_maskmem):
            prev_frame_idx = frame_idx - (self.num_maskmem - t_pos)
            t_pos_and_prevs.append((t_pos, non_cond_outputs.get(prev_frame_idx)))

        for t_pos, prev in t_pos_and_prevs:
            if prev is None:
                continue
            feats = prev["maskmem_features"].to(device, non_blocking=True)
            to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
            pos = prev["maskmem_pos_enc"][-1].to(device).flatten(2).permute(2, 0, 1)
            to_cat_memory_pos.append(
                pos + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
            )

        # DIVERGENCE 3: the horizon is clamped by frames actually seen, and
        # t_diff_max normalises by that same clamped value. Upstream uses
        # min(num_frames, max_obj_ptrs_in_encoder) - 1, where num_frames is the
        # whole clip length -- unknowable when streaming. Early in a sequence the
        # two differ and every pointer's temporal phase shifts.
        num_obj_ptr_tokens = 0
        n_ptrs = min(frame_idx + 1, self.max_obj_ptrs_in_encoder)
        pos_and_ptrs = [
            (frame_idx - t, out["obj_ptr"])
            for t, out in cond_outputs.items()
            if t <= frame_idx
        ]
        for t_diff in range(1, n_ptrs):
            out = non_cond_outputs.get(frame_idx - t_diff)
            if out is not None:
                pos_and_ptrs.append((t_diff, out["obj_ptr"]))

        if pos_and_ptrs:
            pos_list, ptrs_list = zip(*pos_and_ptrs)
            obj_ptrs = torch.stack(ptrs_list, dim=0)
            t_diff_max = n_ptrs - 1
            obj_pos = torch.tensor(pos_list, dtype=torch.float32, device=device)
            obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=C)  # local copy!
            obj_pos = self.obj_ptr_tpos_proj(obj_pos)
            obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
            if self.mem_dim < C:
                # a 256-d pointer becomes four 64-d memory tokens
                ptr_repeat = C // self.mem_dim
                obj_ptrs = obj_ptrs.reshape(-1, B, ptr_repeat, self.mem_dim)
                obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                obj_pos = (
                    obj_pos.unsqueeze(1)
                    .expand(-1, ptr_repeat, -1, -1)
                    .reshape(-1, B, self.mem_dim)
                )
            to_cat_memory.append(obj_ptrs)
            to_cat_memory_pos.append(obj_pos)
            num_obj_ptr_tokens = obj_ptrs.shape[0]

        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos = torch.cat(to_cat_memory_pos, dim=0)
        out = self.memory_attention(
            curr=current_vision_feats[-1],
            curr_pos=current_vision_pos_embeds[-1],
            memory=memory,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        return out.permute(1, 2, 0).view(B, C, H, W)


def get_1d_sine_pe(pos_inds: torch.Tensor, dim: int, temperature: float = 10000.0):
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = torch.exp((2 * (dim_t // 2) / pe_dim) * math.log(float(temperature)))
    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    return torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
