"""
ImageEnhancerV2
===============
An improved lightweight CNN for restoration + 2x super-resolution.

Improvements over V1:
    * wider feature extraction
    * residual blocks with skip connections
    * a global skip connection around the body
    * PixelShuffle upsampling for clean 2x scaling

Pipeline:
    [B, 3, 128, 128]
        -> initial conv
        -> residual blocks
        -> PixelShuffle (2x)
        -> refine conv
        -> + global skip of the bicubic-upscaled input
    [B, 3, 256, 256]
"""

import torch.nn as nn
import torch.nn.functional as F

from .residual_block import ResidualBlock


class ImageEnhancerV2(nn.Module):
    def __init__(self, in_channels=3, mid_channels=48, out_channels=3,
                 num_res_blocks=4):
        super().__init__()
        self.in_channels = in_channels

        # Initial feature extraction at 128x128
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Residual backbone (feature refinement)
        blocks = [ResidualBlock(mid_channels) for _ in range(num_res_blocks)]
        self.backbone = nn.Sequential(*blocks)

        self.tail = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)

        # 2x upsampling: mid_channels -> 4*out_channels, then PixelShuffle
        self.upscale = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
        )

        # Refinement at 256x256
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # Global skip path: bicubic upsample the input to 256x256.
        # This preserves strong structural information.
        if x.size(-1) != 256:
            skip = F.interpolate(x, scale_factor=2, mode="bicubic",
                                 align_corners=False, recompute_scale_factor=False)
        else:
            skip = x

        h = self.head(x)          # [B, mid, 128, 128]
        h = self.backbone(h)      # residual refinement
        h = self.tail(h)
        up = self.upscale(h)      # [B, out, 256, 256]

        out = self.refine(up)
        out = out + skip          # global residual connection
        return out
