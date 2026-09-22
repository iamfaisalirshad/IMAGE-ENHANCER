"""
Degradation pipeline
====================
Turns a clean 256x256 image into a degraded 128x128 image.

Steps (configurable, see config.py):
    1. Gaussian blur
    2. Downsample to 128x128 (bicubic / area)
    3. Gaussian noise
    (optional) JPEG compression, brightness, contrast
"""

import cv2
import numpy as np
import torch

from config import BLUR_SIGMA, NOISE_LEVEL, JPEG_QUALITY, DO_JPEG


def _to_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def degrade_array(clean, sigma=None, noise=None, target_size=128,
                  do_jpeg=None, jpeg_quality=None):
    """
    Degrade a clean RGB array (H, W, 3) in 0..255 into a
    128x128 low-quality array. Returns (degraded, clean_resized, degraded_rgb)
    All arrays are float32 in 0..255.
    """
    if sigma is None:
        sigma = BLUR_SIGMA
    if noise is None:
        noise = NOISE_LEVEL
    if do_jpeg is None:
        do_jpeg = DO_JPEG
    if jpeg_quality is None:
        jpeg_quality = JPEG_QUALITY

    img = _to_uint8(clean)

    # 1. Gaussian blur
    if sigma and sigma > 0:
        k = int(round(sigma * 6)) | 1  # odd kernel size
        k = max(3, k)
        img = cv2.GaussianBlur(img, (k, k), sigma)

    # 2. Downsample to target_size
    h, w = img.shape[:2]
    if (h, w) != (target_size, target_size):
        degraded = cv2.resize(img, (target_size, target_size),
                              interpolation=cv2.INTER_AREA)
    else:
        degraded = img.copy()

    # 3. Optional JPEG compression
    if do_jpeg:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        ok, buf = cv2.imencode(".jpg", degraded, encode_param)
        if ok:
            degraded = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # 4. Noise
    if noise and noise > 0:
        rng = np.random.default_rng()
        noise_arr = rng.normal(0, noise * 255.0, degraded.shape)
        degraded = degraded.astype(np.float32) + noise_arr
        degraded = _to_uint8(degraded)

    return degraded.astype(np.float32)


def degrade_tensor(img_tensor, target_size=128, sigma=None, noise=None):
    """
    img_tensor: tensor [B, 3, H, W] normalized to [0,1].
    sigma/noise may be given to override the config defaults.
    Returns a degraded tensor [B, 3, target, target] in [0,1].
    """
    img_tensor = img_tensor.detach().cpu()
    outs = []
    for i in range(img_tensor.size(0)):
        arr = img_tensor[i].permute(1, 2, 0).numpy() * 255.0
        degraded = degrade_array(arr, target_size=target_size,
                                 sigma=sigma, noise=noise)
        t = torch.from_numpy(degraded / 255.0).permute(2, 0, 1).float()
        outs.append(t)
    return torch.stack(outs)
