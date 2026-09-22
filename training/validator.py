"""Validation: compute mean loss over the validation set."""

import torch


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Return (mean_loss, mean_psnr) over the validation set."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    count = 0

    for low, target in val_loader:
        low = low.to(device)
        target = target.to(device)
        pred = model(low)
        loss = criterion(pred, target)
        total_loss += loss.item() * low.size(0)

        # PSNR on 0..1 tensors
        mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
        psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
        total_psnr += psnr.sum().item()
        count += low.size(0)

    model.train()
    if count == 0:
        return 0.0, 0.0
    return total_loss / count, total_psnr / count
