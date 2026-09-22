"""
Dataset + DataLoader factory.
=================================
Loads full-resolution (>=256) images from a folder, takes a random
256x256 crop as the clean target, and produces a degraded 128x128 input
on the fly (blur + downsample + noise, with randomized strength each
time so the model generalises better).
"""

import random

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import (BATCH_SIZE, NUM_WORKERS, IMAGE_SIZE, TARGET_SIZE,
                    BLUR_SIGMA_MIN, BLUR_SIGMA_MAX, NOISE_MIN, NOISE_MAX)
from utils.file_utils import find_images

from .degradation import degrade_tensor


class ImageEnhancerDataset(Dataset):
    """Clean 256x256 targets + on-the-fly degraded 128x128 inputs."""

    def __init__(self, folder, limit=None):
        self.paths = find_images(folder, limit=limit)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        tw, th = TARGET_SIZE, TARGET_SIZE

        if image.width >= tw and image.height >= th:
            # random crop -> more variety per epoch even with few files
            left = random.randint(0, image.width - tw)
            top = random.randint(0, image.height - th)
            image = image.crop((left, top, left + tw, top + th))
        else:
            image = image.resize((tw, th), Image.BICUBIC)

        target = self.to_tensor(image)          # [3, 256, 256] in [0,1]

        # Degraded input from the target, using randomised strength
        sigma = random.uniform(BLUR_SIGMA_MIN, BLUR_SIGMA_MAX)
        noise = random.uniform(NOISE_MIN, NOISE_MAX)
        low = degrade_tensor(target.unsqueeze(0).float(),
                             target_size=IMAGE_SIZE,
                             sigma=sigma, noise=noise)[0]   # [3,128,128]
        return low, target


def make_dataloaders(train_dir, val_dir, batch_size=BATCH_SIZE,
                     num_workers=NUM_WORKERS, limit=None,
                     train_limit=None, val_limit=None):
    """
    Build train and val DataLoaders.
    Returns (train_loader, val_loader, summary_dict).
    If a split has zero images, its loader is None (never construct a
    DataLoader over an empty dataset, which would raise).
    """
    train_set = ImageEnhancerDataset(train_dir, limit=train_limit or limit)
    val_set = ImageEnhancerDataset(val_dir, limit=val_limit or limit)

    train_loader = None
    val_loader = None
    if len(train_set) > 0:
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=(len(train_set) > batch_size),
        )
    if len(val_set) > 0:
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
        )

    summary = {
        "train_images": len(train_set),
        "val_images": len(val_set),
        "train_batches": len(train_loader) if train_loader else 0,
        "val_batches": len(val_loader) if val_loader else 0,
    }
    return train_loader, val_loader, summary