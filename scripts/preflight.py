#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import torch

from fdmrnet.config import load_config
from fdmrnet.degradation import ThroughPlaneDegrader
from fdmrnet.losses import CompositeFDMRNetLoss
from fdmrnet.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch = tuple(cfg["data"]["patch_size_dhw"])
    hr = torch.rand((1, 1, *patch), device=device)
    degrader = ThroughPlaneDegrader(**{k: v for k, v in cfg["degradation"].items() if k != "axis"})
    lr, target, meta = degrader.degrade(hr)
    model = build_model(cfg["model"]).to(device).train()
    outputs = model(lr, patch)
    batch = {"hr": target, "brain_mask": torch.ones_like(target), "field_target": torch.zeros((1, 3, *patch), device=device)}
    loss, components = CompositeFDMRNetLoss(**cfg["loss"])(outputs, batch)
    loss.backward()
    finite = bool(torch.isfinite(loss).item()) and all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters()
    )
    report = {
        "device": str(device),
        "hr_shape": list(hr.shape),
        "lr_shape": list(lr.shape),
        "output_shape": list(outputs["pred"].shape),
        "degradation": meta.to_dict(),
        "loss": float(loss.item()),
        "components": {key: float(value.item()) for key, value in components.items()},
        "finite": finite,
        "parameters": sum(p.numel() for p in model.parameters()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0,
    }
    print(json.dumps(report, indent=2))
    if not finite or outputs["pred"].shape != target.shape:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

