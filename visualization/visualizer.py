"""
Visualization helpers.
Produces training curves and side-by-side comparison grids.
All figures are saved (never shown interactively) so the app
can run headless.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import CURVE_DIR, COMPARISON_DIR
from utils.file_utils import ensure_dirs, save_result_image


def _save_fig(fig, out_dir, name):
    ensure_dirs([out_dir])
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_history(histories, out_dir=CURVE_DIR):
    """
    `histories`: list of (model_name, history_dict).
    Saves loss curves per model and an optional combined PSNR line plot.
    Returns a list of saved paths.
    """
    paths = []
    for name, hist in histories:
        fig, ax = plt.subplots(figsize=(7, 4))
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax.plot(epochs, hist["train_loss"], label="Train Loss", marker="o")
        ax.plot(epochs, hist["val_loss"], label="Val Loss", marker="s")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{name.upper()} Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = _save_fig(fig, out_dir, f"loss_{name.lower()}.png")
        paths.append(p)

    return paths


def plot_metric_bars(results, out_dir=CURVE_DIR, metric="psnr"):
    """
    `results`: list of dicts like evaluator output.
    metric: 'psnr' or 'ssim'. Produces a grouped bar chart
    comparing Bicubic vs each model.
    """
    labels = ["Bicubic"]
    model_vals = {"Bicubic": []}
    bic_key = f"avg_{metric}_bicubic"
    model_key = f"avg_{metric}_model"

    for r in results:
        labels.append(r["model"].upper())
        model_vals.setdefault(r["model"].upper(), [])
        model_vals["Bicubic"].append(r.get(bic_key, 0))
        model_vals[r["model"].upper()].append(r.get(model_key, 0))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    width = 0.3
    for i, label in enumerate(labels):
        vals = model_vals[label]
        ax.bar(x + i * width, vals, width, label=label)
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Val Set 1"] * len(x))
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} Comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    p = _save_fig(fig, out_dir, f"{metric}_comparison.png")
    return p


def _to_pil(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def save_comparison_grid(low, low_resized, bicubic, pred, target,
                         out_dir=COMPARISON_DIR, name=None, titles=None):
    """
    Build a side-by-side grid:
        Degraded | Bicubic | Model | Ground Truth
    Returns the saved path.
    """
    tiles = [low_resized, bicubic, pred, target]
    titles = titles or ["Degraded Input", "Bicubic", "Model Output",
                        "Ground Truth"]

    cols = len(tiles)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    for ax, tile, title in zip(axes, tiles, titles):
        ax.imshow(_to_pil(tile))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()

    ensure_dirs([out_dir])
    path = os.path.join(out_dir, (name or "comparison_001") + ".png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
