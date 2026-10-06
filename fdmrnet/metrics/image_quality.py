from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean()
    m = mask.to(dtype=x.dtype)
    if m.shape != x.shape:
        m = m.expand_as(x)
    return (x * m).sum() / m.sum().clamp_min(1)


def psnr_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None, data_range: float = 1.0) -> torch.Tensor:
    mse = _masked_mean((pred - target).square(), mask)
    return 10 * torch.log10(torch.as_tensor(data_range**2, device=pred.device, dtype=pred.dtype) / mse.clamp_min(1e-12))


def _gaussian3d(window: int, sigma: float, device, dtype) -> torch.Tensor:
    radius = window // 2
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = g / g.sum()
    kernel = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return kernel.view(1, 1, window, window, window)


def ssim3d_masked(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    data_range: float = 1.0,
    window: int = 7,
    sigma: float = 1.5,
) -> torch.Tensor:
    kernel = _gaussian3d(window, sigma, pred.device, pred.dtype)
    pad = window // 2
    mu_x = F.conv3d(pred, kernel, padding=pad)
    mu_y = F.conv3d(target, kernel, padding=pad)
    sigma_x = F.conv3d(pred.square(), kernel, padding=pad) - mu_x.square()
    sigma_y = F.conv3d(target.square(), kernel, padding=pad) - mu_y.square()
    sigma_xy = F.conv3d(pred * target, kernel, padding=pad) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1e-12)
    return _masked_mean(ssim, mask)


def nmse_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        numerator, denominator = (pred - target).square().sum(), target.square().sum()
    else:
        m = mask.to(dtype=pred.dtype)
        numerator = ((pred - target).square() * m).sum()
        denominator = (target.square() * m).sum()
    return numerator / denominator.clamp_min(1e-12)


def hfen_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    # Discrete 3D Laplacian provides a deterministic HF error measure.
    kernel = torch.zeros((1, 1, 3, 3, 3), device=pred.device, dtype=pred.dtype)
    kernel[0, 0, 1, 1, 1] = -6
    kernel[0, 0, 0, 1, 1] = kernel[0, 0, 2, 1, 1] = 1
    kernel[0, 0, 1, 0, 1] = kernel[0, 0, 1, 2, 1] = 1
    kernel[0, 0, 1, 1, 0] = kernel[0, 0, 1, 1, 2] = 1
    error = F.conv3d(pred - target, kernel, padding=1).square()
    reference = F.conv3d(target, kernel, padding=1).square()
    return torch.sqrt(_masked_mean(error, mask) / _masked_mean(reference, mask).clamp_min(1e-12))


def metric_bundle(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None, data_range: float = 1.0) -> dict[str, float]:
    valid_n = int(mask.sum().item()) if mask is not None else int(target.numel())
    if valid_n == 0:
        return {"psnr": math.nan, "ssim": math.nan, "nmse": math.nan, "hfen": math.nan, "valid_n": 0}
    pred = pred.clamp(0, data_range)
    target = target.clamp(0, data_range)
    return {
        "psnr": float(psnr_masked(pred, target, mask, data_range).item()),
        "ssim": float(ssim3d_masked(pred, target, mask, data_range).item()),
        "nmse": float(nmse_masked(pred, target, mask).item()),
        "hfen": float(hfen_masked(pred, target, mask).item()),
        "valid_n": valid_n,
    }

