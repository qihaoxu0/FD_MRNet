from __future__ import annotations

import math
import torch


def complete_amp_optimizer_step(*, scaler, optimizer, scheduler, parameters, gradient_clip: float) -> dict:
    """Complete one AMP optimizer boundary using GradScaler's native overflow semantics."""
    params = list(parameters)
    old_scale = float(scaler.get_scale())
    scaler.unscale_(optimizer)
    first_nonfinite = None
    finite_norm_sq = 0.0
    for name, parameter in params:
        if parameter.grad is None:
            continue
        finite = bool(torch.isfinite(parameter.grad).all().item())
        if not finite and first_nonfinite is None:
            first_nonfinite = {
                "name": name,
                "nan": int(torch.isnan(parameter.grad).sum().item()),
                "inf": int(torch.isinf(parameter.grad).sum().item()),
            }
        elif finite:
            norm = float(torch.linalg.vector_norm(parameter.grad.detach().float()).item())
            finite_norm_sq += norm * norm
    found = scaler._found_inf_per_device(optimizer) if scaler.is_enabled() else {}
    overflow = any(float(value.item()) != 0.0 for value in found.values())
    if overflow != (first_nonfinite is not None):
        raise RuntimeError("GradScaler found_inf disagrees with explicit gradient finite check")
    if overflow:
        grad_norm = math.inf
    else:
        clipped = torch.nn.utils.clip_grad_norm_([parameter for _, parameter in params], gradient_clip)
        grad_norm = float(clipped.item())
        if not math.isfinite(grad_norm):
            raise FloatingPointError("Finite gradients produced a non-finite gradient norm")
    scaler.step(optimizer)
    scaler.update()
    new_scale = float(scaler.get_scale())
    if overflow:
        if new_scale >= old_scale:
            raise RuntimeError("GradScaler detected overflow but did not reduce its scale")
        optimizer_updated = False
    else:
        if scheduler is not None:
            scheduler.step()
        optimizer_updated = True
    return {
        "overflow": overflow,
        "optimizer_updated": optimizer_updated,
        "old_scale": old_scale,
        "new_scale": new_scale,
        "gradient_norm": grad_norm,
        "finite_gradient_norm": math.sqrt(finite_norm_sq),
        "first_nonfinite_parameter": first_nonfinite,
    }
