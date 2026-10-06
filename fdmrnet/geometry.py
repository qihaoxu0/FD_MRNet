from __future__ import annotations

import torch
import torch.nn.functional as F


def warp_volume(x: torch.Tensor, disp_vox_dhw: torch.Tensor, padding_mode: str = "border") -> torch.Tensor:
    """Warp BCDHW tensor with voxel displacement channels ordered D,H,W."""
    if x.ndim != 5 or disp_vox_dhw.ndim != 5 or disp_vox_dhw.shape[1] != 3:
        raise ValueError("Expected x=B,C,D,H,W and displacement=B,3,D,H,W")
    b, _, d, h, w = x.shape
    z = torch.linspace(-1, 1, d, device=x.device, dtype=x.dtype)
    y = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
    xx = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
    zz, yy, grid_x = torch.meshgrid(z, y, xx, indexing="ij")
    base = torch.stack((grid_x, yy, zz), dim=-1).unsqueeze(0).expand(b, -1, -1, -1, -1)
    dz, dy, dx = disp_vox_dhw.unbind(dim=1)
    norm_x = 2 * dx / max(w - 1, 1)
    norm_y = 2 * dy / max(h - 1, 1)
    norm_z = 2 * dz / max(d - 1, 1)
    delta = torch.stack((norm_x, norm_y, norm_z), dim=-1)
    return F.grid_sample(x, base + delta, mode="bilinear", padding_mode=padding_mode, align_corners=True)


def smooth_random_field(
    batch: int,
    shape_dhw: tuple[int, int, int],
    max_displacement: float,
    coarse_grid: tuple[int, int, int] = (4, 8, 8),
    translation: float = 1.0,
    *,
    device=None,
    dtype=None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    random = torch.randn((batch, 3, *coarse_grid), device=device, dtype=dtype, generator=generator)
    field = F.interpolate(random, size=shape_dhw, mode="trilinear", align_corners=True)
    field = field / field.flatten(2).std(dim=2, keepdim=True).clamp_min(1e-6).view(batch, 3, 1, 1, 1)
    field = torch.tanh(field) * max_displacement
    shift = (torch.rand((batch, 3, 1, 1, 1), device=device, dtype=dtype, generator=generator) * 2 - 1)
    return field + shift * translation


def upsample_displacement(field: torch.Tensor, shape_dhw: tuple[int, int, int]) -> torch.Tensor:
    return resize_displacement(field, shape_dhw)


def resize_displacement(field: torch.Tensor, shape_dhw: tuple[int, int, int]) -> torch.Tensor:
    """Resize a voxel-unit displacement field while preserving physical displacement."""
    old = field.shape[-3:]
    out = F.interpolate(field, size=shape_dhw, mode="trilinear", align_corners=True)
    scale = torch.tensor(
        [shape_dhw[0] / old[0], shape_dhw[1] / old[1], shape_dhw[2] / old[2]],
        device=field.device,
        dtype=field.dtype,
    ).view(1, 3, 1, 1, 1)
    return out * scale


def invert_displacement(field: torch.Tensor, iterations: int = 7) -> torch.Tensor:
    """Fixed-point inverse for the pull-warp convention used by ``warp_volume``.

    If ``corrupted = warp_volume(clean, inverse)`` then warping ``corrupted`` by
    ``field`` approximately recovers ``clean``. Small, smooth fields converge rapidly.
    """
    inverse = -field
    for _ in range(int(iterations)):
        inverse = -warp_volume(field, inverse)
    return inverse


def spatial_gradients(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dd = field[:, :, 1:] - field[:, :, :-1]
    dh = field[:, :, :, 1:] - field[:, :, :, :-1]
    dw = field[:, :, :, :, 1:] - field[:, :, :, :, :-1]
    return dd, dh, dw


def jacobian_determinant(field: torch.Tensor) -> torch.Tensor:
    """Finite-difference det(I + grad(u)); output cropped to common interior."""
    u = field[:, :, :-1, :-1, :-1]
    d_d = field[:, :, 1:, :-1, :-1] - u
    d_h = field[:, :, :-1, 1:, :-1] - u
    d_w = field[:, :, :-1, :-1, 1:] - u
    j11, j21, j31 = 1 + d_d[:, 0], d_d[:, 1], d_d[:, 2]
    j12, j22, j32 = d_h[:, 0], 1 + d_h[:, 1], d_h[:, 2]
    j13, j23, j33 = d_w[:, 0], d_w[:, 1], 1 + d_w[:, 2]
    return (
        j11 * (j22 * j33 - j23 * j32)
        - j12 * (j21 * j33 - j23 * j31)
        + j13 * (j21 * j32 - j22 * j31)
    )
