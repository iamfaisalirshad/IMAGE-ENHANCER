"""Device detection: CUDA if available, otherwise CPU."""

import torch

from config import DEVICE_CPU_LABEL, DEVICE_CUDA_LABEL


def get_device():
    """
    Return the best available torch device.
    Prefers CUDA when present, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_summary():
    """Return a human readable device string."""
    if get_device().type == "cuda":
        name = torch.cuda.get_device_name(0)
        return f"{DEVICE_CUDA_LABEL} ({name})"
    return DEVICE_CPU_LABEL
