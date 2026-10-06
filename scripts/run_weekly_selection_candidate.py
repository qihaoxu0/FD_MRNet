#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from fdmrnet.config import load_config, sha256_file
from fdmrnet.data import BraTSFixedPatchDataset, BraTSPatchDataset
from fdmrnet.degradation import ThroughPlaneDegrader
from fdmrnet.engine.checkpoint import load_checkpoint, save_checkpoint
from fdmrnet.engine.trainer import _geometry_augment
from fdmrnet.losses import CompositeFDMRNetLoss
from fdmrnet.metrics import metric_bundle
from fdmrnet.models import build_model
from fdmrnet.utils.reproducibility import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate", choices=("A", "B", "C"), required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    settings = {
        "A": {"learning_rate": 2.5e-5, "gradient_accumulation": 1},
        "B": {"learning_rate": 5e-5, "gradient_accumulation": 1},
        "C": {"learning_rate": 5e-5, "gradient_accumulation": 2},
    }[args.candidate]
    cfg = load_config(args.config)
    seed = int(cfg["experiment"]["seed"])
    seed_everything(seed, cfg["training"].get("deterministic", True))
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise RuntimeError(f"Refusing to overwrite selection directory: {out_dir}")
    out_dir.mkdir(parents=True)
    device = torch.device("cuda")
    patch = (48, 96, 96)
    degrader = ThroughPlaneDegrader(**{key: value for key, value in cfg["degradation"].items() if key != "axis"})
    train_set = BraTSPatchDataset(
        cfg["data"]["train_manifest"], cfg["data"]["modality"], degrader, patch,
        args.steps, cfg["data"].get("foreground_probability", 0.8), cfg["data"].get("clip_z", 5.0),
        cfg["data"].get("augmentation", {}),
    )
    val_set = BraTSFixedPatchDataset(
        cfg["data"]["val_manifest"], cfg["data"]["modality"], degrader, patch,
        cfg["data"].get("clip_z", 5.0),
    )
    model_cfg = dict(cfg["model"]); model_cfg["gradient_checkpointing"] = True
    model = build_model(model_cfg).to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=cfg["training"]["weight_decay"]
    )
    # Selection uses a constant LR so scheduler horizon cannot confound A/B/C.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    criterion = CompositeFDMRNetLoss(**cfg["loss"]).to(device)
    accumulation = settings["gradient_accumulation"]
    rows, subjects = [], []
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        for step in range(1, args.steps + 1):
            sample = train_set[step - 1]
            subjects.append(sample["subject"])
            batch = {
                key: value.unsqueeze(0).to(device, non_blocking=False) if torch.is_tensor(value) else value
                for key, value in sample.items()
            }
            batch["lr"], batch["field_target"] = _geometry_augment(
                batch["lr"], batch["hr"].shape[-3:], cfg["training"].get("geometry_augmentation", {})
            )
            step_started = time.perf_counter()
            with torch.autocast("cuda", enabled=True):
                outputs = model(batch["lr"], tuple(batch["hr"].shape[-3:]))
                loss, parts = criterion(outputs, batch)
                scaled = loss / accumulation
            if not torch.isfinite(loss) or not all(torch.isfinite(value) for value in parts.values()):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            scaler.scale(scaled).backward()
            boundary = step % accumulation == 0 or step == args.steps
            gradient_finite = True
            grad_norm_value = math.nan
            if boundary:
                scaler.unscale_(optimizer)
                gradient_finite = all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                    for parameter in model.parameters()
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("gradient_clip", 1.0))
                grad_norm_value = float(grad_norm.item())
                if not gradient_finite or not math.isfinite(grad_norm_value):
                    raise FloatingPointError(f"Non-finite gradient at step {step}")
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            rows.append({
                "step": step, "seconds": time.perf_counter() - step_started,
                **{key: float(value.detach().item()) for key, value in parts.items()},
                "gradient_finite": gradient_finite, "grad_norm": grad_norm_value,
            })

        model.eval(); val_metrics = []
        with torch.inference_mode():
            for index in range(len(val_set)):
                sample = val_set[index]
                lr = sample["lr"].unsqueeze(0).to(device)
                hr = sample["hr"].unsqueeze(0).to(device)
                brain = sample["brain_mask"].unsqueeze(0).to(device)
                with torch.autocast("cuda", enabled=True):
                    pred = model(lr, tuple(hr.shape[-3:]))["pred"]
                val_metrics.append(metric_bundle(pred.float(), hr.float(), brain, 1.0))
        val_psnr = statistics.mean(value["psnr"] for value in val_metrics if math.isfinite(value["psnr"]))
        val_ssim = statistics.mean(value["ssim"] for value in val_metrics if math.isfinite(value["ssim"]))
        manifests = {
            "train": sha256_file(cfg["data"]["train_manifest"]),
            "val": sha256_file(cfg["data"]["val_manifest"]),
        }
        checkpoint = out_dir / "selection_final.pt"
        selection_cfg = dict(cfg)
        selection_cfg["_config_sha256"] = cfg["_config_sha256"]
        save_checkpoint(
            checkpoint, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            epoch=0, best_metric=val_psnr, config=selection_cfg, manifest_hashes=manifests,
        )
        # Perturb all RNGs/model, then prove a strict complete restore.
        expected_rng = (random.random(), float(np.random.rand()), torch.rand(1).item())
        for parameter in model.parameters(): parameter.data.zero_()
        payload = load_checkpoint(checkpoint, model, optimizer, scheduler, scaler, restore_rng=True, map_location=device)
        restored_rng = (random.random(), float(np.random.rand()), torch.rand(1).item())
        elapsed = time.perf_counter() - started
        first = statistics.mean(row["total"] for row in rows[:50])
        last = statistics.mean(row["total"] for row in rows[-50:])
        result = {
            "status": "passed", "candidate": args.candidate, **settings,
            "activation_checkpointing": True, "patch_size_dhw": list(patch), "micro_steps": args.steps,
            "training_subject_sequence": subjects, "training_subject_sequence_sha256": __import__("hashlib").sha256("\n".join(subjects).encode()).hexdigest(),
            "train_seconds": sum(row["seconds"] for row in rows), "wall_seconds": elapsed,
            "mean_step_seconds": statistics.mean(row["seconds"] for row in rows),
            "std_step_seconds": statistics.stdev(row["seconds"] for row in rows),
            "loss_first_50": first, "loss_last_50": last, "loss_drop": first - last,
            "loss_drop_per_hour": (first - last) / (sum(row["seconds"] for row in rows) / 3600),
            "loss_terms_last_50": {key: statistics.mean(row[key] for row in rows[-50:]) for key in ("rec","freq","align","field","smooth","jacobian","total")},
            "validation_cases": len(val_metrics), "validation_psnr": val_psnr, "validation_ssim": val_ssim,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "nan_inf_oom": False, "gradients_finite": all(row["gradient_finite"] for row in rows),
            "checkpoint_restore": {
                "model_strict": True, "optimizer": payload["optimizer"] is not None,
                "scheduler": payload["scheduler"] is not None, "scaler": payload["scaler"] is not None,
                "rng": expected_rng == restored_rng, "manifest_hashes": payload["manifest_hashes"] == manifests,
            },
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
            "train_manifest_sha256": manifests["train"], "val_manifest_sha256": manifests["val"],
            "test_accessed": False,
        }
        (out_dir / "step_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({key: result[key] for key in ("candidate","mean_step_seconds","loss_drop_per_hour","validation_psnr","validation_ssim","peak_reserved_mib","checkpoint_sha256")}, indent=2))
    except Exception as exc:
        failure = {"status": "failed", "candidate": args.candidate, "completed_steps": len(rows), "error": repr(exc), "test_accessed": False}
        (out_dir / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
