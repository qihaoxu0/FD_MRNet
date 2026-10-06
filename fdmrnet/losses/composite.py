from __future__ import annotations

import torch
import torch.nn as nn

from fdmrnet.degradation.gaussian import blur_depth
from fdmrnet.geometry import jacobian_determinant, spatial_gradients, warp_volume


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    error = (pred - target).abs()
    if mask is None:
        return error.mean()
    mask = mask.to(dtype=error.dtype)
    return (error * mask).sum() / mask.sum().clamp_min(1)


class CompositeFDMRNetLoss(nn.Module):
    """Reconstruction plus non-trivial alignment/field regularization.

    The invalid identity objective ||W(X,D)-X|| is deliberately absent.
    """

    def __init__(
        self,
        frequency_sigma: float = 1.0,
        lambda_rec: float = 1.0,
        lambda_freq: float = 0.2,
        lambda_align: float = 0.1,
        lambda_field: float = 0.05,
        lambda_smooth: float = 0.01,
        lambda_jacobian: float = 0.01,
    ) -> None:
        super().__init__()
        self.frequency_sigma = float(frequency_sigma)
        self.weights = {
            "rec": float(lambda_rec),
            "freq": float(lambda_freq),
            "align": float(lambda_align),
            "field": float(lambda_field),
            "smooth": float(lambda_smooth),
            "jacobian": float(lambda_jacobian),
        }

    def forward(self, outputs: dict, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred, target = outputs["pred"], batch["hr"]
        mask = batch.get("brain_mask")
        target_low = blur_depth(target, self.frequency_sigma, "replicate")
        target_high = target - target_low
        rec = masked_l1(pred, target, mask)
        freq = masked_l1(outputs["pred_low"], target_low, mask) + masked_l1(outputs["pred_high"], target_high, mask)
        aligned_coarse = warp_volume(outputs["coarse"], outputs["d_composite"])
        align = masked_l1(aligned_coarse, target_low, mask)
        composite_all = outputs["d_low"] + outputs["d_high"]
        if "field_target" in batch:
            field = (composite_all - batch["field_target"].unsqueeze(1)).abs().mean()
        else:
            field = pred.new_zeros(())
        # Displacement finite differences and determinant products are numerically
        # sensitive under AMP. Keep this complete path in FP32 while preserving the
        # graph so gradients still propagate to the mixed-precision model outputs.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            field_flat = composite_all.flatten(0, 1).float()
            dd, dh, dw = spatial_gradients(field_flat)
            smooth = (dd.square().mean() + dh.square().mean() + dw.square().mean()) / 3
            jac = jacobian_determinant(field_flat)
            jacobian = torch.relu(-jac).mean()
        components = {
            "rec": rec,
            "freq": freq,
            "align": align,
            "field": field,
            "smooth": smooth,
            "jacobian": jacobian,
        }
        total = sum(self.weights[key] * value for key, value in components.items())
        components["total"] = total
        return total, components
