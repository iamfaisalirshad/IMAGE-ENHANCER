"""
Training engine
===============
Trains a model, records history, and saves the single best
validation checkpoint.
"""

import os
import time

import torch
from tqdm import tqdm

from .losses import build_criterion
from .validator import validate


def _div(state_dict):
    """Move a state dict off GPU and clone it for CPU-hosted saving."""
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def train_model(model, model_name, train_loader, val_loader, device,
                epochs, learning_rate, checkpoint_path, quit_flag=None,
                seed=0):
    """
    Train `model` for `epochs` and save only the best validation model.
    Returns a history dict with per-epoch:
        train_loss, val_loss, val_psnr, time, best
    `quit_flag` may be a reference to a dict with {'quit': bool} so the
    GUI/main loop can stop training gracefully.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Sharper perceptual loss: L1 + SSIM + high-frequency detail.
    # This makes the model restore crisp edges instead of a smooth blur,
    # which is what you can actually SEE after enhancement.
    criterion = build_criterion()

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "time": [],
        "best": [],
    }
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        if quit_flag is not None and quit_flag.get("quit", False):
            print("\n[Training interrupted by user.]")
            break

        model.train()
        running = 0.0
        start = time.time()
        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch}/{epochs} [train]",
                    leave=False)
        for low, target in pbar:
            low = low.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = model(low)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            running += loss.item() * low.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / max(len(train_loader.dataset), 1)
        val_loss, val_psnr = validate(model, val_loader, criterion, device)
        elapsed = time.time() - start

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                "model_state": _div(model.state_dict()),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_psnr": val_psnr,
                "model_name": model_name,
                "seed": seed,
            }, checkpoint_path)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["time"].append(elapsed)
        history["best"].append(is_best)

        print(f"  Epoch {epoch:2d}/{epochs} | "
              f"train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | "
              f"val_psnr {val_psnr:.2f} dB | "
              f"{'BEST' if is_best else ''} | {elapsed:.1f}s")

    return history
