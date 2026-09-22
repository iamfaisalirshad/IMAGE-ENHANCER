"""Reproducibility helpers."""

import os
import random

import numpy as np
import torch


def set_seed(seed):
    """
    Set the random seed for Python, NumPy, and PyTorch
    so runs are reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make operations deterministic where possible (may slow CPU a little)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
