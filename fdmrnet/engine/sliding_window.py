from __future__ import annotations

import itertools
import math

import torch


def _starts(length: int, patch: int, overlap: int) -> list[int]:
    if patch >= length:
        return [0]
    stride = patch - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than patch")
    values = list(range(0, max(1, length - patch + 1), stride))
    if values[-1] != length - patch:
        values.append(length - patch)
    return values


def gaussian_importance_map(shape_dhw: tuple[int, int, int], sigma_scale: float, device, dtype) -> torch.Tensor:
    axes = []
    for n in shape_dhw:
        x = torch.arange(n, device=device, dtype=dtype)
        center = (n - 1) / 2
        sigma = max(n * sigma_scale, 1e-3)
        axes.append(torch.exp(-0.5 * ((x - center) / sigma) ** 2))
    weight = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return weight.clamp_min(1e-6).unsqueeze(0).unsqueeze(0)


@torch.inference_mode()
def sliding_window_predict(
    model,
    coarse_volume: torch.Tensor,
    patch_size_dhw: tuple[int, int, int],
    overlap_dhw: tuple[int, int, int],
    sigma_scale: float = 0.125,
    amp: bool = True,
) -> tuple[torch.Tensor, dict]:
    if coarse_volume.shape[0] != 1:
        raise ValueError("Full-volume inference currently requires batch size 1")
    _, _, d, h, w = coarse_volume.shape
    pd, ph, pw = patch_size_dhw
    if pd > d or ph > h or pw > w:
        padded = torch.nn.functional.pad(
            coarse_volume,
            (0, max(0, pw - w), 0, max(0, ph - h), 0, max(0, pd - d)),
            mode="replicate",
        )
    else:
        padded = coarse_volume
    dp, hp, wp = padded.shape[-3:]
    starts = [_starts(n, p, o) for n, p, o in zip((dp, hp, wp), patch_size_dhw, overlap_dhw)]
    weight = gaussian_importance_map(patch_size_dhw, sigma_scale, padded.device, padded.dtype)
    output = torch.zeros_like(padded)
    weights = torch.zeros_like(padded)
    model.eval()
    patch_count = 0
    device_type = padded.device.type
    for sd, sh, sw in itertools.product(*starts):
        patch = padded[:, :, sd : sd + pd, sh : sh + ph, sw : sw + pw]
        with torch.autocast(device_type=device_type, enabled=amp and device_type == "cuda"):
            pred = model(patch, patch_size_dhw)["pred"]
        output[:, :, sd : sd + pd, sh : sh + ph, sw : sw + pw] += pred.float() * weight.float()
        weights[:, :, sd : sd + pd, sh : sh + ph, sw : sw + pw] += weight.float()
        patch_count += 1
    output = output / weights.clamp_min(1e-8)
    seam_ratio = float((weights.max() / weights.min().clamp_min(1e-8)).item())
    return output[:, :, :d, :h, :w], {
        "patches": patch_count,
        "coverage_min": float(weights.min().item()),
        "coverage_max": float(weights.max().item()),
        "seam_ratio": seam_ratio,
        "padded_shape_dhw": [dp, hp, wp],
    }

