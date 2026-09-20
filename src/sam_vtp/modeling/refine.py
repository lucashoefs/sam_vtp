import torch
from torch import nn


class RefinementHead(nn.Module):
    """Full-resolution RGB-guided residual refinement of the coarse mask."""

    def __init__(self, channels: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(4, channels, kernel_size=3, padding=1),
            _FusedGroupNorm(1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.out = nn.Conv2d(channels, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.to(memory_format=torch.channels_last)

    def forward(
        self, coarse_logits: torch.Tensor, image_norm: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([image_norm, coarse_logits], dim=1)
        # The model hands this head NCHW, so the conversion is real work, but it
        # measured 0.06 ms against the 7.5 ms the convolutions save.
        x = x.contiguous(memory_format=torch.channels_last)
        # `out` has one channel, where NHWC and NCHW are the same bytes, so the
        # residual add needs no conversion back.
        return coarse_logits + self.out(self.stem(x))


class _FusedGroupNorm(nn.GroupNorm):
    """GroupNorm computed as one reduction and one affine, for `num_groups=1`."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_groups != 1:
            return super().forward(x)
        dims = tuple(range(1, x.dim()))
        shape = (1, -1) + (1,) * (x.dim() - 2)
        # Reductions accumulate in fp32 even for a half input, so a bank of
        # 2.1M values does not lose the mean to rounding.
        var, mean = torch.var_mean(x, dim=dims, keepdim=True, unbiased=False)
        scale = self.weight.view(shape) * torch.rsqrt(var + self.eps)
        return torch.addcmul(self.bias.view(shape) - mean * scale, x, scale)
