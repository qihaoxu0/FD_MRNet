from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch


def load_nifti_dhw(path: str | Path, dtype=np.float32) -> tuple[torch.Tensor, np.ndarray]:
    image = nib.load(str(path))
    array = np.asarray(image.dataobj, dtype=dtype)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI: {path}, got {array.shape}")
    array = np.ascontiguousarray(array.transpose(2, 0, 1))
    return torch.from_numpy(array), image.affine.copy()


def normalize_nonzero(volume: torch.Tensor, clip_z: float = 5.0) -> tuple[torch.Tensor, torch.Tensor]:
    mask = volume != 0
    if mask.sum() == 0:
        raise ValueError("Empty MRI volume")
    values = volume[mask].float()
    mean = values.mean()
    std = values.std().clamp_min(1e-6)
    normalized = ((volume.float() - mean) / std).clamp(-clip_z, clip_z)
    normalized = (normalized + clip_z) / (2 * clip_z)
    normalized = torch.where(mask, normalized, torch.zeros_like(normalized))
    return normalized, mask

