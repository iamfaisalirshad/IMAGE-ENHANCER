"""
Evaluation runner: evaluates a model against Bicubic on the
validation set and stores per-image + aggregate metrics as JSON.
"""

import json
import os

import torch
from tqdm import tqdm

from config import METRIC_DIR
from utils.file_utils import ensure_dirs

from .metrics import evaluate_tensor_pair, psnr, ssim, bicubic_upscale


def evaluate_model(model, val_loader, device, model_name, save=True):
    """
    Returns a dict:
        {
            "model": model_name,
            "images": N,
            "avg_psnr_bicubic": float,
            "avg_ssim_bicubic": float,
            "avg_psnr_model": float,
            "avg_ssim_model": float,
        }
    Accepts either a raw torch Module or an inference.ImageEnhancerModel
    wrapper (it transparently unwraps the wrapper).
    """
    from inference.predictor import ImageEnhancerModel
    if isinstance(model, ImageEnhancerModel):
        model = model.model

    model.eval()
    agg = {"psnr_bicubic": 0.0, "ssim_bicubic": 0.0,
           "psnr_model": 0.0, "ssim_model": 0.0}
    count = 0

    for low, target in tqdm(val_loader, desc=f"Evaluating {model_name}",
                            leave=False):
        low_d = low.to(device)
        target_d = target.to(device)

        with torch.no_grad():
            pred = model(low_d)
        bic = bicubic_upscale(low_d.cpu(), out_size=target.shape[-1]).cpu()

        p_bic = psnr(bic, target)
        s_bic = ssim(bic, target)
        p_model = psnr(pred.cpu(), target)
        s_model = ssim(pred.cpu(), target)

        agg["psnr_bicubic"] += p_bic
        agg["ssim_bicubic"] += s_bic
        agg["psnr_model"] += p_model
        agg["ssim_model"] += s_model
        count += 1

    if count == 0:
        return {"model": model_name, "images": 0}

    result = {
        "model": model_name,
        "images": count,
        "avg_psnr_bicubic": round(agg["psnr_bicubic"] / count, 4),
        "avg_ssim_bicubic": round(agg["ssim_bicubic"] / count, 4),
        "avg_psnr_model": round(agg["psnr_model"] / count, 4),
        "avg_ssim_model": round(agg["ssim_model"] / count, 4),
    }

    if save:
        ensure_dirs([METRIC_DIR])
        path = os.path.join(METRIC_DIR, f"metrics_{model_name.lower()}.json")
        with open(path, "w") as fh:
            json.dump(result, fh, indent=2)
        result["saved_to"] = path

    return result
