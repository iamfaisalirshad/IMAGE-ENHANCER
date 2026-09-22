"""File / folder helpers used across the project."""

import os


def ensure_dirs(directory_list):
    """
    Create every directory in the list if it does not exist.
    Returns nothing; raises if a path cannot be created.
    """
    for path in directory_list:
        if path:
            os.makedirs(path, exist_ok=True)


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def find_images(folder, limit=None):
    """
    Return a sorted list of image file paths inside `folder`.
    If `limit` is set, return at most that many.
    """
    if not os.path.isdir(folder):
        return []
    files = []
    try:
        for name in sorted(os.listdir(folder)):
            ext = os.path.splitext(name)[1].lower()
            if ext in _IMAGE_EXTS:
                files.append(os.path.join(folder, name))
    except PermissionError:
        return []
    if limit is not None:
        files = files[:limit]
    return files


def _make_relative(path):
    """Store the result path relative to the project root, if possible."""
    from config import BASE_DIR

    try:
        return os.path.relpath(path, BASE_DIR)
    except ValueError:
        return path


def save_result_image(image, out_dir, prefix="result"):
    """
    Save a NumPy image (H, W, 3 in 0..255) to `out_dir` with a
    unique timestamped name. Returns the saved absolute path.
    """
    import time

    import numpy as np
    from PIL import Image

    ensure_dirs([out_dir])
    stamp = time.strftime("%Y%m%d_%H%M%S")
    counter = 0
    while True:
        name = f"{prefix}_{stamp}" + (f"_{counter}" if counter else "") + ".png"
        path = os.path.join(out_dir, name)
        if not os.path.exists(path):
            break
        counter += 1

    arr = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return _make_relative(path)
