"""
Generate synthetic sample images (no data download needed).
Used to populate data/sample/ so the "Quick Demo" training and the
comparison feature work out of the box before any dataset is added.
Images are 512x512 so the 256x256 training targets can be random-cropped
from different regions -> more variety when training on few files.
"""

import os

import numpy as np

from config import SAMPLE_DIR, SAMPLE_COUNT

_SIZE = 512


def _shape(kind, h, w, rng):
    yy, xx = np.mgrid[0:h, 0:w]
    if kind == 0:      # diagonal colour sweep
        return (
            0.35 + 0.45 * np.sin(xx / w * 5 * np.pi),
            0.35 + 0.45 * np.cos(yy / h * 4 * np.pi),
            0.30 + 0.50 * np.sin((xx + yy) / (h + w) * 2 * np.pi),
        )
    if kind == 1:      # radial sweep
        r = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
        return (
            0.40 + 0.40 * np.sin(r / max(w, h) * 8 * np.pi),
            0.40 + 0.40 * np.cos(r / max(w, h) * 6 * np.pi),
            0.30 + 0.40 * np.sin(r / max(w, h) * 10 * np.pi + 1),
        )
    if kind == 2:      # horizontal bands
        return (
            0.35 + 0.40 * np.sin(yy / h * 9 * np.pi),
            0.35 + 0.40 * np.cos(yy / h * 7 * np.pi + 0.5),
            0.30 + 0.45 * np.sin(yy / h * 11 * np.pi),
        )
    # chess-like grating
    return (
        0.40 + 0.35 * np.sign(np.sin(xx / 24.0)) * np.sign(np.sin(yy / 24.0)),
        0.40 + 0.35 * np.sign(np.sin(xx / 18.0 + 1)) * np.sign(np.sin(yy / 18.0)),
        0.40 + 0.35 * np.sign(np.sin((xx + yy) / 20.0)),
    )


def _blend_circles(img, h, w, rng):
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(rng.integers(4, 9)):
        cx = rng.integers(0, w)
        cy = rng.integers(0, h)
        rad = rng.integers(30, 140)
        color = rng.random(3) * 0.7 + 0.15
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < rad ** 2
        img[mask] = img[mask] * 0.65 + color * 0.35
    return img


def _synthetic_image(rng):
    """Draw a colourful, textured, copyright-free image at _SIZE x _SIZE."""
    h = w = _SIZE
    img = np.zeros((h, w, 3), dtype=np.float32)
    kind = int(rng.integers(0, 4))
    r, g, b = _shape(kind, h, w, rng)
    img[..., 0] = r
    img[..., 1] = g
    img[..., 2] = b
    img = _blend_circles(img, h, w, rng)
    img += rng.random((h, w, 3)) * 0.06          # texture
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def generate_samples(count=None, size=_SIZE):
    """Create `count` synthetic images in data/sample/. Returns number made."""
    from PIL import Image

    from utils.file_utils import ensure_dirs

    ensure_dirs([SAMPLE_DIR])
    count = count or SAMPLE_COUNT
    rng = np.random.default_rng(0)
    made = 0
    for i in range(count):
        arr = _synthetic_image(rng)
        path = os.path.join(SAMPLE_DIR, f"sample_{i:03d}.png")
        Image.fromarray(arr).save(path)
        made += 1
    return made


if __name__ == "__main__":
    print(f"Generated {generate_samples()} sample images in {SAMPLE_DIR}")