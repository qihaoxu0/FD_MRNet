from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .gaussian import blur_depth


@dataclass(frozen=True)
class DegradationMeta:
    scale: int
    sigma_vox: float
    original_depth: int
    padded_depth: int
    lr_depth: int
    sample_offset: int
    padding_mode: str
    interpolation: str
    noise_kind: str
    noise_std: float

    def to_dict(self) -> dict:
        return asdict(self)


class ThroughPlaneDegrader:
    """Blur then decimate only the through-plane D axis.

    The input convention is B,C,D,H,W. For formal experiments, sigma is expressed
    in native HR voxels and should be fixed in the YAML configuration.
    """

    def __init__(
        self,
        scale: int,
        sigma_vox: float,
        sample_offset: int | str = "center",
        padding_mode: str = "replicate",
        interpolation: str = "trilinear",
        noise_kind: str = "none",
        noise_std: float = 0.0,
    ) -> None:
        if scale not in (2, 4):
            raise ValueError("Formal protocol permits scale 2 or 4")
        if interpolation != "trilinear":
            raise ValueError("This implementation uses explicit 3D trilinear interpolation")
        self.scale = int(scale)
        self.sigma_vox = float(sigma_vox)
        self.sample_offset = self.scale // 2 if sample_offset == "center" else int(sample_offset)
        if not 0 <= self.sample_offset < self.scale:
            raise ValueError("sample_offset must be in [0, scale)")
        self.padding_mode = padding_mode
        self.interpolation = interpolation
        self.noise_kind = noise_kind.lower()
        self.noise_std = float(noise_std)

    def pad_hr(self, hr: torch.Tensor) -> tuple[torch.Tensor, int]:
        depth = hr.shape[-3]
        target_depth = ((depth + self.scale - 1) // self.scale) * self.scale
        extra = target_depth - depth
        if extra:
            hr = F.pad(hr, (0, 0, 0, 0, 0, extra), mode=self.padding_mode)
        return hr, depth

    def add_noise(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        if self.noise_kind == "none" or self.noise_std <= 0:
            return x
        n1 = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) * self.noise_std
        if self.noise_kind == "gaussian":
            return (x + n1).clamp(0, 1)
        if self.noise_kind == "rician":
            n2 = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) * self.noise_std
            return torch.sqrt((x + n1).square() + n2.square()).clamp(0, 1)
        raise ValueError(f"Unsupported noise_kind: {self.noise_kind}")

    def degrade(
        self, hr: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, DegradationMeta]:
        if hr.ndim == 4:
            hr = hr.unsqueeze(0)
            squeeze = True
        elif hr.ndim == 5:
            squeeze = False
        else:
            raise ValueError(f"Expected CDHW or BCDHW, got {tuple(hr.shape)}")
        padded, original_depth = self.pad_hr(hr)
        blurred = blur_depth(padded, self.sigma_vox, self.padding_mode)
        lr = blurred[:, :, self.sample_offset :: self.scale, :, :]
        expected = padded.shape[-3] // self.scale
        lr = lr[:, :, :expected]
        lr = self.add_noise(lr, generator)
        meta = DegradationMeta(
            scale=self.scale,
            sigma_vox=self.sigma_vox,
            original_depth=original_depth,
            padded_depth=padded.shape[-3],
            lr_depth=lr.shape[-3],
            sample_offset=self.sample_offset,
            padding_mode=self.padding_mode,
            interpolation=self.interpolation,
            noise_kind=self.noise_kind,
            noise_std=self.noise_std,
        )
        if squeeze:
            return lr.squeeze(0), padded.squeeze(0), meta
        return lr, padded, meta

    def coarse(self, lr: torch.Tensor, target_shape_dhw: tuple[int, int, int]) -> torch.Tensor:
        squeeze = lr.ndim == 4
        if squeeze:
            lr = lr.unsqueeze(0)
        out = F.interpolate(lr, size=target_shape_dhw, mode="trilinear", align_corners=False)
        return out.squeeze(0) if squeeze else out

