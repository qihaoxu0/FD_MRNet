#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from fdmrnet.config import load_config
from fdmrnet.data import BraTSPatchDataset
from fdmrnet.degradation import ThroughPlaneDegrader
from fdmrnet.engine.trainer import _geometry_augment
from fdmrnet.losses import CompositeFDMRNetLoss
from fdmrnet.models import build_model
from fdmrnet.utils.reproducibility import seed_everything


def finite_gradients(model: torch.nn.Module) -> bool:
    return all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--patch", nargs=3, type=int, required=True)
    parser.add_argument("--strategy", choices=("A", "B", "C"), required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    patch = tuple(args.patch)
    if any(value % 4 for value in patch):
        raise ValueError("Every patch dimension must be divisible by the 4x4x4 attention window")
    accumulation = 1 if args.strategy == "A" else 4
    checkpointing = args.strategy == "C"
    result = {
        "patch_size_dhw": list(patch), "strategy": args.strategy, "batch_size": 1,
        "gradient_accumulation_steps": accumulation, "activation_checkpointing": checkpointing,
        "warmup_steps": args.warmup, "measured_steps": args.steps, "amp": True,
        "status": "started", "oom": False,
    }
    try:
        seed_everything(int(cfg["experiment"]["seed"]), cfg["training"].get("deterministic", True))
        device = torch.device("cuda")
        degrader = ThroughPlaneDegrader(**{k: v for k, v in cfg["degradation"].items() if k != "axis"})
        dataset = BraTSPatchDataset(
            cfg["data"]["train_manifest"], cfg["data"]["modality"], degrader, patch,
            1, cfg["data"].get("foreground_probability", 0.8), cfg["data"].get("clip_z", 5.0),
            cfg["data"].get("augmentation", {}),
        )
        sample = dataset[0]
        batch = {
            key: value.unsqueeze(0).to(device) if torch.is_tensor(value) else value
            for key, value in sample.items()
        }
        source_lr = batch["lr"]
        model_cfg = dict(cfg["model"])
        model_cfg["gradient_checkpointing"] = checkpointing
        model = build_model(model_cfg).to(device).train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"]
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        criterion = CompositeFDMRNetLoss(**cfg["loss"]).to(device)
        optimizer.zero_grad(set_to_none=True)
        times, loss_rows, gradient_ok = [], [], True
        total_iterations = args.warmup + args.steps
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        for iteration in range(1, total_iterations + 1):
            measured = iteration > args.warmup
            if measured:
                torch.cuda.synchronize()
                started = time.perf_counter()
            batch["lr"], batch["field_target"] = _geometry_augment(
                source_lr, batch["hr"].shape[-3:], cfg["training"].get("geometry_augmentation", {})
            )
            with torch.autocast("cuda", enabled=True):
                outputs = model(batch["lr"], tuple(batch["hr"].shape[-3:]))
                loss, parts = criterion(outputs, batch)
                scaled_loss = loss / accumulation
            finite_loss = bool(torch.isfinite(loss).item()) and all(torch.isfinite(value).item() for value in parts.values())
            if not finite_loss:
                raise FloatingPointError("non-finite loss component")
            scaler.scale(scaled_loss).backward()
            boundary = iteration % accumulation == 0 or iteration == total_iterations
            if boundary:
                scaler.unscale_(optimizer)
                gradient_ok = gradient_ok and finite_gradients(model)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("gradient_clip", 1.0))
                gradient_ok = gradient_ok and bool(torch.isfinite(grad_norm).item())
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if measured:
                torch.cuda.synchronize()
                times.append(time.perf_counter() - started)
                loss_rows.append({key: float(value.detach().item()) for key, value in parts.items()})
        result.update({
            "status": "passed" if gradient_ok else "failed_nonfinite_gradient",
            "subject": sample["subject"],
            "input_shape": list(batch["lr"].shape),
            "target_shape": list(batch["hr"].shape),
            "output_shape": list(outputs["pred"].shape),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "mean_step_seconds": statistics.mean(times),
            "std_step_seconds": statistics.stdev(times) if len(times) > 1 else 0.0,
            "step_seconds": times,
            "loss_terms_mean": {
                key: statistics.mean(row[key] for row in loss_rows) for key in loss_rows[0]
            },
            "loss_terms_last": loss_rows[-1],
            "loss_finite": True,
            "gradients_finite": gradient_ok,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "geometry_augmentation": cfg["training"].get("geometry_augmentation", {}),
        })
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        is_oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
        if not is_oom:
            result.update({"status": "failed_runtime", "error": repr(exc)})
        else:
            result.update({"status": "oom", "oom": True, "error": str(exc)})
        if torch.cuda.is_available():
            result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            result["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
            torch.cuda.empty_cache()
    except Exception as exc:
        result.update({"status": "failed", "error": repr(exc)})
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
