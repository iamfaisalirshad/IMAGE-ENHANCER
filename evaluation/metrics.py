"""
Metrics: PSNR, SSIM, and the bicubic baseline.
All functions work on float tensors in the [0,1] range, shape [B,3,H,W].
"""

import torch
import numpy as np
from skimage.metrics import structural_similarity


def psnr(pred, target, data_range=1.0):
    """Peak Signal-to-Noise Ratio in dB (higher is better)."""
    pred = pred.detach().float()
    target = target.detach().float()
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    value = 10.0 * torch.log10((data_range ** 2) / mse)
    return value.item()


def ssim(pred, target, data_range=1.0):
    """Structural Similarity Index Mean (higher is better).
    Converts per-image to numpy and handles RGB channels with a
    per-channel average to keep memory and speed practical."""
    pred_np = pred.detach().float().clamp(0, 1).cpu().numpy()  # B,H,W,3
    target_np = target.detach().float().clamp(0, 1).cpu().numpy()
    # Move channel to last for skimage
    pred_np = pred_np.transpose(0, 2, 3, 1)
    target_np = target_np.transpose(0, 2, 3, 1)

    vals = []
    for i in range(pred_np.shape[0]):
        s = structural_similarity(
            target_np[i], pred_np[i],
            channel_axis=2, data_range=data_range,
            win_size=7,
        )
        vals.append(s)
    return float(np.mean(vals)) if vals else 0.0


def bicubic_upscale(low, out_size=256):
    """Bicubic upsample of a [B,3,H,W] tensor. Returns tensor [B,3,out,out]."""
    return torch.nn.functional.interpolate(
        low, size=(out_size, out_size), mode="bicubic",
        align_corners=False,
    )


def evaluate_tensor_pair(low, target, model, device):
    """
    Evaluate one batch: returns (psnr_bicubic, ssim_bicubic,
                                 psnr_model, ssim_model).
    `low` / `target` are tensors [B,3,128,128] and [B,3,256,256].
    """
    from .metrics import bicubic_upscale

    low = low.to(device)
    target = target.to(target.device) if target.device != device else target

    # Bicubic baseline
    bicubic = bicubic_upscale(low.cpu(), out_size=target.shape[-1]).to(target.device)
    p_bic = psnr(bicubic, target)
    s_bic = ssim(bicubic, target)

    # Model
    with torch.no_grad():
        pred = model(low).to(target.device)
    p_model = psnr(pred, target)
    s_model = ssim(pred, target)

    return p_bic, s_bic, p_model, s_model
