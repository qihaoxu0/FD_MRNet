from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gaussian_kernel1d(sigma: float, truncate: float = 3.0, *, device=None, dtype=None) -> torch.Tensor:
    if sigma <= 0:
        return torch.ones(1, device=device, dtype=dtype)
    radius = max(1, int(math.ceil(truncate * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype or torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def blur_depth(x: torch.Tensor, sigma: float, padding_mode: str = "replicate") -> torch.Tensor:
    """Apply a fixed 1D Gaussian slice-profile blur along D of BCHWD tensor."""
    if x.ndim != 5:
        raise ValueError(f"Expected BCDHW tensor, got {tuple(x.shape)}")
    if sigma <= 0:
        return x
    kernel = gaussian_kernel1d(sigma, device=x.device, dtype=x.dtype)
    radius = kernel.numel() // 2
    weight = kernel.view(1, 1, -1, 1, 1).repeat(x.shape[1], 1, 1, 1, 1)
    padded = F.pad(x, (0, 0, 0, 0, radius, radius), mode=padding_mode)
    return F.conv3d(padded, weight, groups=x.shape[1])

