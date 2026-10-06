from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from fdmrnet.degradation import ThroughPlaneDegrader

from .discovery import SubjectRecord, read_manifest
from .nifti import load_nifti_dhw, normalize_nonzero


def _crop_or_pad(x: torch.Tensor, size: tuple[int, int, int], start: tuple[int, int, int]) -> torch.Tensor:
    d, h, w = size
    sd, sh, sw = start
    crop = x[..., sd : sd + d, sh : sh + h, sw : sw + w]
    pad_d, pad_h, pad_w = max(0, d - crop.shape[-3]), max(0, h - crop.shape[-2]), max(0, w - crop.shape[-1])
    if pad_d or pad_h or pad_w:
        crop = torch.nn.functional.pad(crop, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0)
    return crop


class _BaseBraTS(Dataset):
    def __init__(self, manifest: str | Path, modality: str, clip_z: float = 5.0,
                 external_eval_without_seg: bool = False) -> None:
        self.records: list[SubjectRecord] = read_manifest(manifest)
        self.modality = modality.lower()
        self.clip_z = float(clip_z)
        self.external_eval_without_seg = bool(external_eval_without_seg)
        missing_seg = [record.subject for record in self.records if not record.seg]
        if missing_seg and not self.external_eval_without_seg:
            raise ValueError(
                "Segmentation is required outside explicit external_eval_without_seg mode; "
                f"missing for {len(missing_seg)} subjects"
            )
        # Per-worker LRU cache avoids re-reading a full NIfTI for every random patch.
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_size = 4

    def _load(self, record: SubjectRecord) -> dict[str, torch.Tensor | str]:
        if record.subject in self._cache:
            item = self._cache.pop(record.subject)
            self._cache[record.subject] = item
            return item
        path = getattr(record, self.modality)
        raw, affine = load_nifti_dhw(path)
        hr, brain = normalize_nonzero(raw, self.clip_z)
        if record.seg:
            seg, seg_affine = load_nifti_dhw(record.seg, dtype="int16")
            if seg.shape != raw.shape or not np.allclose(seg_affine, affine, atol=1e-4):
                raise ValueError(f"Segmentation geometry differs from {self.modality} for {record.subject}")
            tumor = seg > 0
        else:
            tumor = torch.zeros_like(brain)
        item = {
            "subject": record.subject,
            "hr": hr.unsqueeze(0),
            "brain_mask": brain.unsqueeze(0),
            "tumor_mask": tumor.unsqueeze(0),
            "segmentation_available": bool(record.seg),
            "affine": torch.as_tensor(affine, dtype=torch.float64),
        }
        self._cache[record.subject] = item
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return item

    def __len__(self) -> int:
        return len(self.records)


class BraTSPatchDataset(_BaseBraTS):
    def __init__(
        self,
        manifest: str | Path,
        modality: str,
        degrader: ThroughPlaneDegrader,
        patch_size_dhw: tuple[int, int, int],
        samples_per_epoch: int,
        foreground_probability: float = 0.8,
        clip_z: float = 5.0,
        augmentation: dict | None = None,
        deterministic_index_seed: int | None = None,
    ) -> None:
        super().__init__(manifest, modality, clip_z)
        self.degrader = degrader
        self.patch_size = tuple(int(v) for v in patch_size_dhw)
        self.samples_per_epoch = int(samples_per_epoch)
        self.foreground_probability = float(foreground_probability)
        self.augmentation = augmentation or {}
        self.deterministic_index_seed = deterministic_index_seed

    def _augment(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply paired, axis-preserving augmentation before LR simulation."""
        values = list(tensors)
        flip_probability = float(self.augmentation.get("flip_probability", 0.0))
        for dim in (-3, -2, -1):
            if random.random() < flip_probability:
                values = [torch.flip(value, dims=(dim,)) for value in values]
        if random.random() < float(self.augmentation.get("inplane_rot90_probability", 0.0)):
            turns = random.randrange(1, 4)
            values = [torch.rot90(value, turns, dims=(-2, -1)) for value in values]
        return tuple(values)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict:
        python_state = torch_state = None
        if self.deterministic_index_seed is not None:
            python_state = random.getstate()
            torch_state = torch.get_rng_state()
            deterministic_seed = int(self.deterministic_index_seed) + int(index)
            random.seed(deterministic_seed)
            torch.manual_seed(deterministic_seed)
        try:
            return self._getitem_impl()
        finally:
            if python_state is not None and torch_state is not None:
                random.setstate(python_state)
                torch.set_rng_state(torch_state)

    def _getitem_impl(self) -> dict:
        record = random.choice(self.records)
        item = self._load(record)
        hr = item["hr"]
        mask = item["brain_mask"]
        shape = hr.shape[-3:]
        if random.random() < self.foreground_probability and mask.any():
            positions = torch.nonzero(mask[0], as_tuple=False)
            center = positions[random.randrange(len(positions))].tolist()
        else:
            center = [random.randrange(max(1, n)) for n in shape]
        start = tuple(max(0, min(n - p, c - p // 2)) for n, p, c in zip(shape, self.patch_size, center))
        hr_patch = _crop_or_pad(hr, self.patch_size, start)
        brain_patch = _crop_or_pad(mask, self.patch_size, start)
        tumor_patch = _crop_or_pad(item["tumor_mask"], self.patch_size, start)
        hr_patch, brain_patch, tumor_patch = self._augment(hr_patch, brain_patch, tumor_patch)
        lr, target, meta = self.degrader.degrade(hr_patch)
        return {
            "subject": record.subject,
            "lr": lr,
            "hr": target,
            "brain_mask": brain_patch,
            "tumor_mask": tumor_patch,
            "original_depth": meta.original_depth,
            "patch_start_dhw": torch.as_tensor(start, dtype=torch.int64),
            "patch_center_dhw": torch.as_tensor(center, dtype=torch.int64),
        }


class BraTSFullVolumeDataset(_BaseBraTS):
    def __init__(self, manifest, modality, degrader: ThroughPlaneDegrader, clip_z: float = 5.0,
                 external_eval_without_seg: bool = False) -> None:
        super().__init__(manifest, modality, clip_z, external_eval_without_seg)
        self.degrader = degrader

    def __getitem__(self, index: int) -> dict:
        item = dict(self._load(self.records[index]))
        lr, target, meta = self.degrader.degrade(item["hr"])
        item.update({"lr": lr, "hr": target, "degradation": meta.to_dict(), "original_depth": meta.original_depth})
        if target.shape[-3] != item["brain_mask"].shape[-3]:
            extra = target.shape[-3] - item["brain_mask"].shape[-3]
            item["brain_mask"] = torch.nn.functional.pad(item["brain_mask"], (0, 0, 0, 0, 0, extra))
            item["tumor_mask"] = torch.nn.functional.pad(item["tumor_mask"], (0, 0, 0, 0, 0, extra))
        return item


class BraTSFixedPatchDataset(_BaseBraTS):
    """One deterministic foreground-centered diagnostic patch per subject."""

    def __init__(self, manifest, modality, degrader, patch_size_dhw, clip_z: float = 5.0) -> None:
        super().__init__(manifest, modality, clip_z)
        self.degrader = degrader
        self.patch_size = tuple(int(v) for v in patch_size_dhw)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        item = self._load(record)
        positions = torch.nonzero(item["brain_mask"][0], as_tuple=False)
        center = positions.float().mean(0).round().long().tolist()
        shape = item["hr"].shape[-3:]
        start = tuple(max(0, min(n - p, c - p // 2)) for n, p, c in zip(shape, self.patch_size, center))
        hr = _crop_or_pad(item["hr"], self.patch_size, start)
        brain = _crop_or_pad(item["brain_mask"], self.patch_size, start)
        tumor = _crop_or_pad(item["tumor_mask"], self.patch_size, start)
        lr, target, meta = self.degrader.degrade(hr)
        return {"subject": record.subject, "lr": lr, "hr": target, "brain_mask": brain,
                "tumor_mask": tumor, "original_depth": meta.original_depth}
