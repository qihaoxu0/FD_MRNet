from __future__ import annotations

import torch

from fdmrnet.geometry import jacobian_determinant


def geometry_metrics(field: torch.Tensor, target: torch.Tensor | None = None) -> dict[str, float]:
    magnitude = torch.linalg.vector_norm(field, dim=1)
    jac = jacobian_determinant(field)
    values = {
        "disp_mean_vox": float(magnitude.mean().item()),
        "disp_p95_vox": float(torch.quantile(magnitude.flatten(), 0.95).item()),
        "disp_max_vox": float(magnitude.max().item()),
        "jacobian_mean": float(jac.mean().item()),
        "jacobian_std": float(jac.std().item()),
        "folding_percent": float((jac <= 0).float().mean().mul(100).item()),
    }
    if target is not None:
        values["field_epe_vox"] = float(torch.linalg.vector_norm(field - target, dim=1).mean().item())
    return values

