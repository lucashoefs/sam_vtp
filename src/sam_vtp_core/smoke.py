from __future__ import annotations

import argparse

import torch

from .factory import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM-VTP smoke test")
    parser.add_argument("--sam2-cfg", type=str, default="configs/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam-vtp-checkpoint", type=str, default="")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    model = build_model(
        sam2_cfg_path=args.sam2_cfg,
        sam_vtp_checkpoint_path=(args.sam_vtp_checkpoint if args.sam_vtp_checkpoint else None),
        device=device,
        num_maskmem=7,
        freeze_encoder=True,
        freeze_decoder=False,
        multimask=True,
        normalization_mode="0_255",
    )

    x = torch.randint(
        low=0,
        high=256,
        size=(1, 3, int(args.height), int(args.width)),
        device=device,
        dtype=torch.int32,
    ).float()

    with torch.no_grad():
        out = model.forward_frame(x, is_init_frame=True, use_rolling_memory=True)

    print("Output shape:", tuple(out["logits"].shape))
    return 0
