"""
ImageEnhancerV1
===============
A simple lightweight CNN baseline for image restoration.

Pipeline:
    [B, 3, 128, 128]
        -> conv blocks (feature extraction)
        -> PixelShuffle (2x upsampling)
        -> conv
    [B, 3, 256, 256]

It has no residual blocks and only a small feature set,
so it acts as the baseline compared against V2.
"""

import torch.nn as nn
import torch.nn.functional as F


class ImageEnhancerV1(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super().__init__()
        self.in_channels = in_channels

        # Feature extraction at 128x128
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 2x upsampling via PixelShuffle.
        # We produce 4 * out_channels channels at 128x128 then
        # rearrange them into out_channels at 256x256.
        self.upscale = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
        )

        # Final refinement at 256x256
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # Global skip: bicubic-upscale the input to 256x256.
        # This keeps the output anchored to a sensible image even when the
        # network is freshly initialised (a common, light baseline design).
        if x.size(-1) != 256:
            skip = F.interpolate(x, scale_factor=2, mode="bicubic",
                                 align_corners=False, recompute_scale_factor=False)
        else:
            skip = x

        f = self.features(x)
        up = self.upscale(f)
        out = self.refine(up)
        out = out + skip
        return out
