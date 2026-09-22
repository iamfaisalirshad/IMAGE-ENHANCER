"""A simple residual block used by ImageEnhancerV2."""

import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    A two-conv residual block with ReLU activation and a
    skip connection.
        x -> conv -> relu -> conv -> + x -> relu
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        out = out + identity
        out = self.relu2(out)
        return out
