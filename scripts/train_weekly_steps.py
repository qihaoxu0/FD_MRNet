#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from fdmrnet.config import load_config, save_resolved_config, sha256_file
from fdmrnet.data import BraTSFixedPatchDataset, BraTSPatchDataset
from fdmrnet.degradation import ThroughPlaneDegrader
from fdmrnet.engine.checkpoint import load_checkpoint, save_checkpoint
from fdmrnet.engine.amp_step import complete_amp_optimizer_step
from fdmrnet.engine.trainer import _geometry_augment
from fdmrnet.losses import CompositeFDMRNetLoss, masked_l1
from fdmrnet.metrics import metric_bundle
from fdmrnet.models import build_model
from fdmrnet.utils.io import append_csv, atomic_json_dump
from fdmrnet.utils.reproducibility import seed_everything, worker_seed


def validate(model, dataset, device, amp, degrader):
    model.eval(); model_rows, baseline_rows = [], []
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            lr = sample["lr"].unsqueeze(0).to(device)
            hr = sample["hr"].unsqueeze(0).to(device)
            brain = sample["brain_mask"].unsqueeze(0).to(device)
            with torch.autocast("cuda", enabled=amp):
                pred = model(lr, tuple(hr.shape[-3:]))["pred"]
            coarse = degrader.coarse(lr, tuple(hr.shape[-3:]))
            model_rows.append(metric_bundle(pred.float(), hr.float(), brain, 1.0))
            baseline_rows.append(metric_bundle(coarse.float(), hr.float(), brain, 1.0))
    model.train()
    finite = lambda rows, key: [row[key] for row in rows if math.isfinite(row[key])]
    return {
        "model_psnr": sum(finite(model_rows, "psnr")) / len(finite(model_rows, "psnr")),
        "model_ssim": sum(finite(model_rows, "ssim")) / len(finite(model_rows, "ssim")),
        "baseline_psnr": sum(finite(baseline_rows, "psnr")) / len(finite(baseline_rows, "psnr")),
        "baseline_ssim": sum(finite(baseline_rows, "ssim")) / len(finite(baseline_rows, "ssim")),
        "cases": len(model_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    cfg = load_config(args.config); tcfg = cfg["training"]; dcfg = cfg["data"]
    deadline = json.loads(Path(args.deadline).read_text(encoding="utf-8-sig"))
    seed_everything(int(cfg["experiment"]["seed"]), tcfg.get("deterministic", True))
    device = torch.device("cuda"); amp = bool(tcfg.get("amp", True))
    output = Path(cfg["experiment"]["output_dir"])
    if output.exists() and not args.resume:
        raise RuntimeError(f"Refusing to overwrite existing run: {output}")
    output.mkdir(parents=True, exist_ok=True); save_resolved_config(cfg, output / "config.yaml")
    degrader = ThroughPlaneDegrader(**{key: value for key, value in cfg["degradation"].items() if key != "axis"})
    maximum = int(tcfg["maximum_micro_steps"]); minimum = int(tcfg["minimum_micro_steps"])
    patch = tuple(dcfg["patch_size_dhw"])
    train_set = BraTSPatchDataset(
        dcfg["train_manifest"], dcfg["modality"], degrader, patch, maximum,
        dcfg.get("foreground_probability", 0.8), dcfg.get("clip_z", 5.0), dcfg.get("augmentation", {}),
        deterministic_index_seed=int(cfg["experiment"]["seed"]) * 100000,
    )
    val_set = BraTSFixedPatchDataset(dcfg["val_manifest"], dcfg["modality"], degrader, patch, dcfg.get("clip_z", 5.0))
    model = build_model(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    accumulation = int(tcfg["gradient_accumulation"])
    optimizer_steps_max = math.ceil(maximum / accumulation)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimizer_steps_max)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = CompositeFDMRNetLoss(**cfg["loss"]).to(device)
    manifests = {"train": sha256_file(dcfg["train_manifest"]), "val": sha256_file(dcfg["val_manifest"])}
    start_step, best, stale = 1, -math.inf, 0
    successful_optimizer_steps = 0; skipped_optimizer_steps = 0
    consecutive_overflows = 0; overflow_steps: list[int] = []
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, restore_rng=True, map_location=device)
        if payload.get("config_sha256") != cfg["_config_sha256"] or payload.get("manifest_hashes") != manifests:
            raise RuntimeError("Resume refused: config or train/val manifest hash changed")
        start_step = int(payload["epoch"]) + 1; best = float(payload["best_metric"])
        runtime = payload.get("config", {}).get("weekly_runtime_state", {})
        stale = int(runtime.get("stale_validations", 0))
        successful_optimizer_steps = int(runtime.get("successful_optimizer_steps", payload["scheduler"]["last_epoch"]))
        skipped_optimizer_steps = int(runtime.get("skipped_optimizer_steps", 0))
        consecutive_overflows = int(runtime.get("consecutive_overflows", 0))
        overflow_steps = [int(value) for value in runtime.get("overflow_steps", [])]
    subset = Subset(train_set, range(start_step - 1, maximum))
    loader = DataLoader(
        subset, batch_size=1, shuffle=False, num_workers=int(dcfg.get("num_workers", 4)),
        pin_memory=bool(tcfg.get("pin_memory", True)), persistent_workers=bool(tcfg.get("persistent_workers", True)),
        worker_init_fn=worker_seed,
    )
    validation_interval = int(tcfg["validation_interval_micro_steps"])
    checkpoint_interval = int(tcfg["checkpoint_interval_micro_steps"])
    patience = int(tcfg["early_stopping_patience"]); min_delta = float(tcfg["early_stopping_min_delta_psnr"])
    completed = start_step - 1; optimizer.zero_grad(set_to_none=True); model.train()
    task_started = time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)
    try:
        for offset, batch in enumerate(loader):
            step = start_step + offset
            batch = {key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value) for key, value in batch.items()}
            batch["lr"], batch["field_target"] = _geometry_augment(batch["lr"], batch["hr"].shape[-3:], tcfg.get("geometry_augmentation", {}))
            step_started = time.perf_counter()
            with torch.autocast("cuda", enabled=amp):
                outputs = model(batch["lr"], tuple(batch["hr"].shape[-3:]))
                if "pred_low" in outputs: loss, parts = criterion(outputs, batch)
                else:
                    loss = masked_l1(outputs["pred"], batch["hr"], batch.get("brain_mask")); parts = {"total": loss, "rec": loss}
                scaled = loss / accumulation
            if not torch.isfinite(loss) or not all(torch.isfinite(value) for value in parts.values()):
                raise FloatingPointError(f"Non-finite loss at micro-step {step}")
            scaler.scale(scaled).backward()
            boundary = step % accumulation == 0 or step == maximum
            grad_norm_value = math.nan
            amp_result = None
            if boundary:
                amp_result = complete_amp_optimizer_step(
                    scaler=scaler, optimizer=optimizer, scheduler=scheduler,
                    parameters=model.named_parameters(), gradient_clip=tcfg.get("gradient_clip", 1.0),
                )
                grad_norm_value = amp_result["gradient_norm"]
                if amp_result["overflow"]:
                    skipped_optimizer_steps += 1; consecutive_overflows += 1; overflow_steps.append(step)
                    recent_overflows = [value for value in overflow_steps if value >= step - 499]
                    append_csv(output / "amp_overflow_events.csv", {
                        "micro_step": step, "subject": batch["subject"][0],
                        "patch_start_dhw": batch["patch_start_dhw"][0].tolist(),
                        "patch_center_dhw": batch["patch_center_dhw"][0].tolist(),
                        "old_scale": amp_result["old_scale"], "new_scale": amp_result["new_scale"],
                        "first_nonfinite_parameter": amp_result["first_nonfinite_parameter"]["name"],
                        "first_nonfinite_nan": amp_result["first_nonfinite_parameter"]["nan"],
                        "first_nonfinite_inf": amp_result["first_nonfinite_parameter"]["inf"],
                        "consecutive_overflows": consecutive_overflows, "overflows_in_last_500": len(recent_overflows),
                    })
                    if consecutive_overflows >= 3:
                        raise FloatingPointError(f"Three consecutive AMP overflows at micro-step {step}")
                    if len(recent_overflows) > 5:
                        raise FloatingPointError(f"More than five AMP overflows in 500 micro-steps at {step}")
                    if amp_result["new_scale"] < 8192:
                        raise FloatingPointError(f"AMP overflow reduced scale below 8192 at micro-step {step}")
                else:
                    successful_optimizer_steps += 1; consecutive_overflows = 0
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); completed = step
            append_csv(output / "micro_step_metrics.csv", {
                "micro_step": step, **{f"loss_{key}": float(value.detach().item()) for key, value in parts.items()},
                "gradient_norm": grad_norm_value, "learning_rate": optimizer.param_groups[0]["lr"],
                "step_seconds": time.perf_counter() - step_started,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            })
            append_csv(output / "amp_step_metrics.csv", {
                "micro_step": step, "successful_optimizer_steps": successful_optimizer_steps,
                "skipped_optimizer_steps": skipped_optimizer_steps,
                "optimizer_updated": bool(amp_result and amp_result["optimizer_updated"]),
                "overflow": bool(amp_result and amp_result["overflow"]),
                "old_scale": amp_result["old_scale"] if amp_result else scaler.get_scale(),
                "new_scale": amp_result["new_scale"] if amp_result else scaler.get_scale(),
            })
            should_validate = step % validation_interval == 0
            if should_validate:
                values = validate(model, val_set, device, amp, degrader)
                append_csv(output / "validation_metrics.csv", {"micro_step": step, **values})
                improved = values["model_psnr"] >= best + min_delta
                if improved: best, stale = values["model_psnr"], 0
                else: stale += 1
                checkpoint_cfg = dict(cfg); checkpoint_cfg["weekly_runtime_state"] = {
                    "stale_validations": stale,
                    "successful_optimizer_steps": successful_optimizer_steps,
                    "skipped_optimizer_steps": skipped_optimizer_steps,
                    "consecutive_overflows": consecutive_overflows,
                    "overflow_steps": overflow_steps,
                }
                common = dict(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=step,
                              best_metric=best, config=checkpoint_cfg, manifest_hashes=manifests)
                save_checkpoint(output / "latest.pt", **common)
                if improved: save_checkpoint(output / "best.pt", **common)
                if step % checkpoint_interval == 0: save_checkpoint(output / f"step_{step:06d}.pt", **common)
                gate = {"micro_step": step, **values, "best_validation_psnr": best, "stale_validations": stale,
                        "checkpoint_restore_verified": False, "successful_optimizer_steps": successful_optimizer_steps,
                        "skipped_optimizer_steps": skipped_optimizer_steps, "amp_scale": scaler.get_scale()}
                # Verify a strict full restore without perturbing the live trainer.
                shadow = build_model(cfg["model"]).to(device)
                shadow_opt = torch.optim.AdamW(shadow.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
                shadow_sched = torch.optim.lr_scheduler.CosineAnnealingLR(shadow_opt, T_max=optimizer_steps_max)
                shadow_scaler = torch.amp.GradScaler("cuda", enabled=amp)
                check = load_checkpoint(output / "latest.pt", shadow, shadow_opt, shadow_sched, shadow_scaler, restore_rng=False, map_location=device)
                gate["checkpoint_restore_verified"] = check["epoch"] == step
                del shadow, shadow_opt, shadow_sched, shadow_scaler; torch.cuda.empty_cache()
                atomic_json_dump(gate, output / f"validation_step_{step:06d}.json")
                atomic_json_dump({"status":"running", "run_id":cfg["experiment"]["id"], "current_micro_step":step,
                                  "successful_optimizer_steps":successful_optimizer_steps,
                                  "skipped_optimizer_steps":skipped_optimizer_steps, "amp_scale":scaler.get_scale(),
                                  "best_validation_psnr":best, "output_dir":str(output), "training_cutoff_at":deadline["training_cutoff_at"]}, args.status)
                if step >= minimum and stale >= patience: break
        atomic_json_dump({"status":"completed", "completed_micro_steps":completed, "best_validation_psnr":best,
                          "runtime_seconds":time.perf_counter()-task_started}, output / "task_completed.json")
    except Exception as exc:
        atomic_json_dump({"status":"safe_stopped_error", "completed_micro_steps":completed, "error":repr(exc),
                          "traceback":traceback.format_exc()}, output / "task_error.json")
        raise


if __name__ == "__main__":
    main()
