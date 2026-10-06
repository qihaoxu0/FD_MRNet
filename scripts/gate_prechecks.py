#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from fdmrnet.config import load_config, sha256_file
from fdmrnet.data import BraTSFixedPatchDataset
from fdmrnet.degradation import ThroughPlaneDegrader
from fdmrnet.engine.checkpoint import load_checkpoint, save_checkpoint
from fdmrnet.losses import CompositeFDMRNetLoss
from fdmrnet.models import build_model
from fdmrnet.utils.reproducibility import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overfit-steps", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["experiment"]["seed"]), cfg["training"].get("deterministic", True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    degrader = ThroughPlaneDegrader(**{k: v for k, v in cfg["degradation"].items() if k != "axis"})
    dataset = BraTSFixedPatchDataset(
        cfg["data"]["train_manifest"], cfg["data"]["modality"], degrader,
        tuple(cfg["data"]["patch_size_dhw"]), cfg["data"].get("clip_z", 5.0),
    )
    sample = dataset[0]
    batch = {
        key: value.unsqueeze(0).to(device) if torch.is_tensor(value) else value
        for key, value in sample.items()
    }
    batch["field_target"] = torch.zeros((1, 3, *batch["hr"].shape[-3:]), device=device)
    model = build_model(cfg["model"]).to(device).train()
    criterion = CompositeFDMRNetLoss(**cfg["loss"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["training"]["epochs"]))
    amp = bool(cfg["training"].get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    losses, components, step_seconds = [], [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.overfit_steps):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            outputs = model(batch["lr"], tuple(batch["hr"].shape[-3:]))
            loss, parts = criterion(outputs, batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss in single-patch overfit")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("gradient_clip", 1.0))
        if not torch.isfinite(grad_norm):
            raise RuntimeError("Non-finite gradient in single-patch overfit")
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        losses.append(float(loss.item()))
        components.append({key: float(value.item()) for key, value in parts.items()})
        step_seconds.append(time.perf_counter() - start)
    scheduler.step()

    original_model = copy.deepcopy(model.state_dict())
    original_optimizer = copy.deepcopy(optimizer.state_dict())
    original_scheduler = copy.deepcopy(scheduler.state_dict())
    original_scaler = copy.deepcopy(scaler.state_dict())
    manifests = {
        "train": sha256_file(cfg["data"]["train_manifest"]),
        "val": sha256_file(cfg["data"]["val_manifest"]),
        "test": sha256_file(cfg["data"]["test_manifest"]),
        "external": sha256_file(cfg["data"]["external_eval_manifest"]),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.resolve().parent) as temp_dir:
        checkpoint = Path(temp_dir) / "roundtrip.pt"
        save_checkpoint(
            checkpoint, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            epoch=1, best_metric=12.5, config=cfg, manifest_hashes=manifests,
        )
        expected_rng = (random.random(), float(np.random.rand()), torch.rand(1).item())
        for parameter in model.parameters():
            parameter.data.zero_()
        payload = load_checkpoint(checkpoint, model, optimizer, scheduler, scaler, restore_rng=True, map_location=device)
        restored_rng = (random.random(), float(np.random.rand()), torch.rand(1).item())
        model_restored = all(torch.equal(model.state_dict()[key].cpu(), value.cpu()) for key, value in original_model.items())
        optimizer_restored = optimizer.state_dict()["param_groups"] == original_optimizer["param_groups"]
        scheduler_restored = scheduler.state_dict() == original_scheduler
        scaler_restored = scaler.state_dict() == original_scaler
    report = {
        "status": "passed",
        "subject": sample["subject"],
        "device": str(device),
        "hr_shape": list(batch["hr"].shape),
        "lr_shape": list(batch["lr"].shape),
        "output_shape": list(outputs["pred"].shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "overfit_steps": args.overfit_steps,
        "losses": losses,
        "components": components,
        "loss_decreased": losses[-1] < losses[0],
        "step_seconds": step_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
        "checkpoint_roundtrip": {
            "epoch": payload["epoch"],
            "model": model_restored,
            "optimizer": optimizer_restored,
            "scheduler": scheduler_restored,
            "scaler": scaler_restored,
            "rng": restored_rng == expected_rng,
            "manifest_hashes": payload["manifest_hashes"] == manifests,
        },
    }
    required = [report["loss_decreased"], *report["checkpoint_roundtrip"].values()]
    if not all(required):
        report["status"] = "failed"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
