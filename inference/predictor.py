"""
Predictor: load a trained checkpoint and produce true 2x super-resolution
output. Arbitrarily-sized images are processed in 128x128 tiles (overlapping,
with seam blending) and each tile is upscaled to 256x256 by the CNN, so the
final image is exactly 2x the input size and nothing is downscaled-then-shown.
"""

import os

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms
from tqdm import tqdm

import config
from config import CHECKPOINT_V1, CHECKPOINT_V2, IMAGE_SIZE, TARGET_SIZE
from models.cnn_v1 import ImageEnhancerV1
from models.cnn_v2 import ImageEnhancerV2
from utils.sharpen import sharpen_pil

MODEL_FACTORY = {
    "v1": (ImageEnhancerV1, CHECKPOINT_V1),
    "v2": (ImageEnhancerV2, CHECKPOINT_V2),
}


class ImageEnhancerModel:
    """Wrapper around a torch model with a friendly API."""

    def __init__(self, model, name, device):
        self.model = model
        self.name = name
        self.device = device
        self.model.to(device).eval()

    @torch.no_grad()
    def infer_tensor(self, low_tensor):
        """low_tensor: [B,3,128,128] in [0,1] -> [B,3,256,256]."""
        return self.model(low_tensor.to(self.device))

    def _infer_patch(self, patch):
        """
        patch: uint8 RGB [128,128,3] -> uint8 RGB [256,256,3].
        The tile is lightly softened first (matching the CNN's training
        distribution) so the model's deblurring genuinely activates and
        the output comes back crisper than the direct upscale.
        """
        if config.TILE_PRE_BLUR and config.TILE_PRE_BLUR > 0.0:
            patch = np.asarray(
                Image.fromarray(patch).filter(
                    ImageFilter.GaussianBlur(config.TILE_PRE_BLUR)),
                dtype=np.uint8)
        tensor = torch.from_numpy(
            patch.transpose(2, 0, 1).astype(np.float32) / 255.0
        ).unsqueeze(0)
        out = self.infer_tensor(tensor)
        arr = out[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        return (arr * 255.0).astype(np.uint8)

    def enhance_2x(self, pil_image):
        """
        True 2x super-resolution for an arbitrarily-sized RGB PIL image.
        Returns (input_2x, enhanced_2x) as uint8 RGB numpy arrays, both
        exactly 2x the size of `pil_image`.

        When config.ENHANCE_ADAPTIVE is on the input is first measured: if it
        is already clean/sharp the original 2x upscale is returned almost
        unchanged (originality preserved, no sharpening or shade shift), and
        restoration strength scales up only as blur/noise are actually found.
        """
        pil_image = pil_image.convert("RGB")
        w, h = pil_image.size

        # Work on a float array; pad to >= IMAGE_SIZE if the image is tiny.
        src = np.asarray(pil_image, dtype=np.uint8)  # (h, w, 3)
        if w < IMAGE_SIZE or h < IMAGE_SIZE:
            scale = max(IMAGE_SIZE / w, IMAGE_SIZE / h)
            pil_image = pil_image.resize(
                (int(round(w * scale)), int(round(h * scale))), Image.BICUBIC)
            w, h = pil_image.size
            src = np.asarray(pil_image, dtype=np.uint8)

        overlap = 8
        stride = IMAGE_SIZE - overlap
        out_w, out_h = w * 2, h * 2
        canvas = np.zeros((out_h, out_w, 3), dtype=np.float64)
        weight = np.zeros((out_h, out_w, 1), dtype=np.float64)

        rows = list(range(0, h - IMAGE_SIZE + 1, stride))
        if rows[-1] != h - IMAGE_SIZE:
            rows.append(h - IMAGE_SIZE)
        cols = list(range(0, w - IMAGE_SIZE + 1, stride))
        if cols[-1] != w - IMAGE_SIZE:
            cols.append(w - IMAGE_SIZE)

        total = len(rows) * len(cols)
        with tqdm(total=total, desc="Enhancing tiles", leave=False) as pbar:
            for r in rows:
                for c in cols:
                    patch = self._crop_patch(src, c, r)
                    up = self._infer_patch(patch)          # 256x256x3
                    up64 = up.astype(np.float64)
                    # weight map with linear ramps across the overlap borders
                    wm = np.ones((256, 256), dtype=np.float64)
                    wm[:2 * overlap] = np.linspace(0, 1, 2 * overlap)[:, None] \
                        if r > 0 else wm[:2 * overlap]
                    wm[-2 * overlap:] = np.linspace(1, 0, 2 * overlap)[:, None] \
                        if r + IMAGE_SIZE < h else wm[-2 * overlap:]
                    wm[:, :2 * overlap] *= np.linspace(0, 1, 2 * overlap)[None, :] \
                        if c > 0 else 1.0
                    wm[:, -2 * overlap:] *= np.linspace(1, 0, 2 * overlap)[None, :] \
                        if c + IMAGE_SIZE < w else 1.0

                    x0, y0 = c * 2, r * 2
                    canvas[y0:y0 + 256, x0:x0 + 256] += up64 * wm[:, :, None]
                    weight[y0:y0 + 256, x0:x0 + 256] += wm[:, :, None]
                    pbar.update(1)

        enhanced = canvas / np.maximum(weight, 1e-8)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        input_2x = np.asarray(
            pil_image.resize((out_w, out_h), Image.BICUBIC), dtype=np.uint8)

        if config.ENHANCE_ADAPTIVE:
            deficiency = self._assess_input(src)
            shade = self._assess_shade(src)
            if (deficiency < config.ADAPTIVE_KEEP_IF_CLEAN and
                    shade < config.SHADE_RESTORE_KEEP_IF_OK):
                # Already clean: keep the original upscale entirely.
                enhanced = input_2x.copy()
            else:
                if deficiency < config.ADAPTIVE_KEEP_IF_CLEAN:
                    # Clean detail but veiled tones: keep the ORIGINAL pixels
                    # untouched and only remove the measured shade below.
                    enhanced = input_2x.astype(np.float64)
                else:
                    # Blend model restoration with the original, scaling the
                    # model share and the sharpening strength with the
                    # measured deficiency, then re-align tones.
                    alpha = min(config.ADAPTIVE_MAX_ALPHA,
                                config.ADAPTIVE_MIN_ALPHA +
                                config.ADAPTIVE_ALPHA_SLOPE * deficiency)
                    enhanced = np.clip(
                        alpha * enhanced.astype(np.float64) +
                        (1.0 - alpha) * input_2x.astype(np.float64),
                        0, 255)
                    amount = config.ENHANCE_SHARPEN_AMOUNT * \
                        (0.40 + 0.75 * deficiency)
                    if amount > 0:
                        enhanced = np.asarray(
                            sharpen_pil(
                                Image.fromarray(
                                    np.clip(enhanced, 0, 255).astype(np.uint8)),
                                amount=amount,
                                radius=config.SHARPEN_RADIUS,
                                threshold=config.SHARPEN_THRESHOLD),
                            dtype=np.float64)
                if shade >= config.SHADE_RESTORE_KEEP_IF_OK:
                    enhanced = self._deveil(enhanced, shade)
                # Re-align tones LAST so the final image keeps exactly the
                # original shades/brightness/colour, never shifted.
                enhanced = self._match_tones(
                    np.clip(enhanced, 0, 255).astype(np.uint8), input_2x)
        else:
            # Fixed full-strength pipeline (legacy behaviour).
            if config.ENHANCE_SHARPEN_AMOUNT and config.ENHANCE_SHARPEN_AMOUNT > 0.0:
                pil_enh = sharpen_pil(
                    Image.fromarray(enhanced),
                    amount=config.ENHANCE_SHARPEN_AMOUNT,
                    radius=config.SHARPEN_RADIUS,
                    threshold=config.SHARPEN_THRESHOLD)
                enhanced = np.asarray(pil_enh, dtype=np.uint8)

        return input_2x, enhanced

    @staticmethod
    def _assess_input(src):
        """
        Measure how blurred/noisy an image really is. Returns a deficiency
        score in [0, 1]:
          0.0 = already clean and sharp (keep the original)
          1.0 = badly blurred or very noisy (restore fully)
        Everything is judged on a 256-normalised grayscale copy so scores are
        comparable across input sizes. Blur is measured as the collapse of
        Laplacian edge energy under extra smoothing (content-independent
        ratio, two noise-robust probes); noise via robust median-MAD of the
        high-frequency residual.
        """
        import cv2
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        maxd = max(gray.shape)
        if maxd > 256:
            s = 256.0 / maxd
            gray = cv2.resize(gray, None, fx=s, fy=s,
                              interpolation=cv2.INTER_AREA)

        g32 = gray.astype(np.float64)

        # Probe 1: original vs. +1.0 blur
        v1 = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        v1b = float(cv2.Laplacian(
            cv2.GaussianBlur(gray, (0, 0), 1.0), cv2.CV_64F).var())
        ratio1 = v1 / max(v1b, 1e-9)

        # Probe 2: lightly denoised (+0.6) vs. +1.6 blur, robust to noise
        g06 = np.uint8(np.clip(cv2.GaussianBlur(gray, (0, 0), 0.6), 0, 255))
        g16 = np.uint8(np.clip(cv2.GaussianBlur(gray, (0, 0), 1.6), 0, 255))
        v2 = float(cv2.Laplacian(g06, cv2.CV_64F).var())
        v2b = float(cv2.Laplacian(g16, cv2.CV_64F).var())
        ratio2 = v2 / max(v2b, 1e-9)

        score1 = (config.BLUR_RATIO_HI - ratio1) / \
            max(config.BLUR_RATIO_HI - config.BLUR_RATIO_LO, 1e-6)
        score2 = (config.BLUR_RATIO2_HI - ratio2) / \
            max(config.BLUR_RATIO2_HI - (config.BLUR_RATIO2_HI / 3.0), 1e-6)
        def_blur = float(np.clip(min(score1, score2), 0.0, 1.0))

        diff = g32 - cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float64)
        noise_est = float(1.4826 * np.median(np.abs(diff)))
        def_noise = float(np.clip(
            (noise_est - config.NOISE_STD_LO) /
            max(config.NOISE_STD_HI - config.NOISE_STD_LO, 1e-6), 0.0, 1.0))

        return float(np.clip(0.7 * def_blur + 0.3 * def_noise, 0.0, 1.0))

    @staticmethod
    def _assess_shade(src):
        """
        Measure how 'washed out / veiled' an image really is. Returns a shade
        score in [0, 1]:
          0.0 = healthy tonal range (nothing to fix)
          1.0 = badly veiled (compressed range + lifted blacks)
        Judged on a 256-normalised grayscale copy so inputs of any size
        compare fairly. A healthy scene with real shadows has a WIDE dynamic
        range (the shadows reach deep), so the range term keeps its shade
        score at ~0 and this never tries to "fix" the photo's shadows.
        """
        import cv2
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        maxd = max(gray.shape)
        if maxd > 256:
            s = 256.0 / maxd
            gray = cv2.resize(gray, None, fx=s, fy=s,
                              interpolation=cv2.INTER_AREA)

        lum = gray.astype(np.float64) / 255.0
        p2, p5, p98 = np.percentile(lum, [2, 5, 98])
        dr = float(p98 - p2)  # luminance pass-band width in [0,1]

        shade_range = float(np.clip(
            (config.SHADE_RANGE_HI - dr) /
            max(config.SHADE_RANGE_HI - config.SHADE_RANGE_LO, 1e-6), 0.0, 1.0))
        shade_black = float(np.clip(
            (p5 - config.SHADE_BLACK_LIFT_LO) /
            max(config.SHADE_BLACK_LIFT_HI - config.SHADE_BLACK_LIFT_LO, 1e-6),
            0.0, 1.0))
        # The black-lift term only counts once the tonal range is genuinely
        # compressed; a normal photo (healthy range, deep blacks) scores ~0.
        return float(np.clip(shade_range * (0.6 + 0.4 * shade_black), 0.0, 1.0))

    @staticmethod
    def _deveil(imgf, shade):
        """
        Remove a washed-out 'shade' while preserving shadows.

        Applies a linear contrast gain (always >= 1, pivoted at the robust
        median luminance) and a small saturation lift scaled by the measured
        `shade`. Because the gain is >= 1 around the median, dark/shadow
        pixels only move darker and can never be lifted; the caller re-aligns
        channel means to the input afterwards, so originality is kept.
        """
        imgf = np.ascontiguousarray(imgf).astype(np.float64)
        lum = 0.299 * imgf[..., 0] + 0.587 * imgf[..., 1] + 0.114 * imgf[..., 2]
        pivot = float(np.median(lum)) if lum.size else 128.0
        gain = 1.0 + config.SHADE_GAIN_MAX * shade
        out = pivot + (imgf - pivot) * gain
        if config.SHADE_SAT_MAX > 0:
            sat = 1.0 + config.SHADE_SAT_MAX * shade
            lum2 = 0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]
            out = lum2[..., None] + (out - lum2[..., None]) * sat
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _match_tones(out, ref):
        """
        Re-align each colour channel's mean to the input so the enhancement
        never shifts the overall shades/brightness of the original image.

        The enhanced pixels sit on the 8-bit integer grid, so sub-integer
        shifts would be evaporated by rounding. We therefore shift each
        channel by the nearest whole gray level, which brings the mean within
        0.5 / 255 (< 0.2%) of the original shades.
        """
        out_f = np.ascontiguousarray(out).astype(np.float64)
        ref_f = np.ascontiguousarray(ref).astype(np.float64)
        for c in range(3):
            offset = int(round(float(ref_f[..., c].mean()) -
                               float(out_f[..., c].mean())))
            if offset:
                out_f[..., c] = out_f[..., c] + offset
        return np.clip(out_f, 0, 255).astype(np.uint8)

    @staticmethod
    def _crop_patch(src, c, r):
        """Return a 128x128 patch at (c, r), reflect-padded at the edges."""
        h, w = src.shape[:2]
        c2 = min(c + IMAGE_SIZE, w)
        r2 = min(r + IMAGE_SIZE, h)
        patch = src[r:r2, c:c2]
        pad_r = IMAGE_SIZE - patch.shape[0]
        pad_c = IMAGE_SIZE - patch.shape[1]
        if pad_r or pad_c:
            patch = np.pad(patch, ((0, pad_r), (0, pad_c), (0, 0)),
                           mode="reflect")
        return patch

    def enhance_single(self, low_array):
        """Single 128x128 input -> 256x256 output (used by eval/compare)."""
        return self._infer_patch(np.asarray(low_array, dtype=np.uint8))


def load_model(name, device, checkpoint_path=None):
    """
    Load the requested model ('v1' or 'v2') from its checkpoint.
    Returns an ImageEnhancerModel, or None if no checkpoint exists.
    """
    cls, default_path = MODEL_FACTORY[name.lower()]
    path = checkpoint_path or default_path
    if not os.path.isfile(path):
        return None

    model = cls()
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    return ImageEnhancerModel(model, name.lower(), device)


def enhance_file(model, image_path):
    """
    Open an image and generate a true 2x enhanced version.
    Returns (input_2x, enhanced_2x) as uint8 RGB arrays (both 2x the
    original image size).
    """
    pil = Image.open(image_path).convert("RGB")
    return model.enhance_2x(pil)