"""
Sharpening helpers
==================
A light, standard unsharp-mask pass applied to the ENHANCED output so the
result looks distinctly crisp. This is a real photo-enhancement technique
(not applied when computing PSNR/SSIM metrics).
"""

from PIL import Image, ImageFilter


def unsharp_mask(image, radius=2, percent=110, threshold=2):
    """
    Apply an unsharp mask to a PIL RGB image and return a new image.
    percent > 100 increases apparent sharpness. Pillow requires ints.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    radius = max(1, int(round(float(radius))))
    percent = max(0, int(round(float(percent))))
    threshold = max(0, int(round(float(threshold))))
    return image.filter(
        ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
    )


def sharpen_pil(image, amount=1.0, radius=2.0, threshold=2):
    """
    Apply unsharp masking scaled by `amount` (0 = none, 1.0 = default).
    Returns an unchanged copy when amount <= 0.02.
    """
    if amount <= 0.02:
        return image
    percent = 110.0 * amount
    return unsharp_mask(image, radius=radius, percent=percent, threshold=threshold)