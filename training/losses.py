"""
Losses for sharper, perceptually-visible enhancement.

The plain L1/MSE loss favours smooth outputs that look like a blurry
bicubic upscale. We add:
    - SSIM loss      (keeps structure/edges, directly improves SSIM metric)
    - Laplacian loss (tries to match high-frequency detail, i.e. edges)
Combined with a small L1 term for stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def ssim_loss(x, y, window_size=11, window_sigma=1.5):
    """1 - mean SSIM between normalized images x and y (torch)."""
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    x = x.clamp(0, 1)
    y = y.clamp(0, 1)

    t = torch.arange(window_size, dtype=x.dtype, device=x.device) - window_size // 2
    g1d = torch.exp(-(t ** 2) / (2 * window_sigma ** 2))
    g1d = g1d / g1d.sum()
    win = g1d[:, None] * g1d[None, :]                 # (ws, ws)
    win = win.expand(x.shape[1], 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu_x = F.conv2d(x, win, padding=pad, groups=x.shape[1])
    mu_y = F.conv2d(y, win, padding=pad, groups=x.shape[1])
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sx2 = F.conv2d(x * x, win, padding=pad, groups=x.shape[1]) - mu_x2
    sy2 = F.conv2d(y * y, win, padding=pad, groups=x.shape[1]) - mu_y2
    sxy = F.conv2d(x * y, win, padding=pad, groups=x.shape[1]) - mu_xy

    num = (2 * mu_xy + c1) * (2 * sxy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sx2 + sy2 + c2)
    return 1.0 - (num / (den + 1e-8)).mean()


def _laplacian_kernel(device, dtype, channels):
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                     dtype=dtype, device=device).reshape(1, 1, 3, 3)
    return k.expand(channels, 1, 3, 3).contiguous()


def high_frequency_loss(x, y):
    """Mean absolute error between Laplacian (high-frequency) maps."""
    lap_k = _laplacian_kernel(x.device, y.dtype, x.shape[1])
    lap_x = F.conv2d(x, lap_k, padding=1, groups=x.shape[1])
    lap_y = F.conv2d(y, lap_k, padding=1, groups=x.shape[1])
    return F.l1_loss(lap_x, lap_y)


def build_criterion(ssim_weight=0.35, hf_weight=0.15):
    """Returns a callable criterion(pred, target)."""
    l1 = nn.L1Loss()

    def criterion(pred, target):
        loss = l1(pred, target)
        loss = loss + ssim_weight * ssim_loss(pred, target)
        loss = loss + hf_weight * high_frequency_loss(pred, target)
        return loss

    return criterion