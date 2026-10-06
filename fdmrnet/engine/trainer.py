from __future__ import annotations

import json
import math
import os
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from fdmrnet.config import save_resolved_config, sha256_file
from fdmrnet.data import BraTSFixedPatchDataset, BraTSPatchDataset
from fdmrnet.degradation import ThroughPlaneDegrader
from fdmrnet.geometry import invert_displacement, resize_displacement, smooth_random_field, warp_volume
from fdmrnet.losses import CompositeFDMRNetLoss, masked_l1
from fdmrnet.metrics import metric_bundle
from fdmrnet.models import build_model
from fdmrnet.utils.io import append_csv, atomic_json_dump
from fdmrnet.utils.reproducibility import seed_everything, worker_seed

from .checkpoint import load_checkpoint, save_checkpoint
from .amp_step import complete_amp_optimizer_step
from .budget import TrainingBudgetLedger


def _distributed() -> tuple[bool, int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    return world > 1, rank, local, world


def _degrader(cfg: dict) -> ThroughPlaneDegrader:
    return ThroughPlaneDegrader(**{k: v for k, v in cfg.items() if k != "axis"})


def _geometry_augment(lr: torch.Tensor, hr_shape, cfg: dict, generator=None):
    if cfg.get("probability", 0) <= 0 or torch.rand((), device=lr.device, generator=generator) > cfg["probability"]:
        return lr, torch.zeros((lr.shape[0], 3, *hr_shape), device=lr.device, dtype=lr.dtype)
    field_hr = smooth_random_field(
        lr.shape[0],
        tuple(hr_shape),
        cfg.get("max_local_displacement_hr_vox", 1.5),
        tuple(cfg.get("coarse_grid_dhw", [4, 8, 8])),
        cfg.get("max_translation_hr_vox", 0.5),
        device=lr.device,
        dtype=lr.dtype,
        generator=generator,
    )
    field_lr = resize_displacement(field_hr, tuple(lr.shape[-3:]))
    corruption_lr = invert_displacement(field_lr, cfg.get("inverse_iterations", 7))
    corrupted = warp_volume(lr, corruption_lr)
    return corrupted, field_hr


def train_from_config(cfg: dict, resume: str | None = None, stop_after_epoch: int | None = None) -> None:
    if cfg["experiment"].get("enabled", True) is False:
        raise RuntimeError(
            "Training disabled by configuration: "
            + str(cfg["experiment"].get("exclusion_reason", "unspecified exclusion"))
        )
    distributed, rank, local_rank, world = _distributed()
    seed = int(cfg["experiment"]["seed"]) + rank
    seed_everything(seed, cfg["training"].get("deterministic", True))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    effective_batch = (
        int(cfg["training"]["batch_size_per_gpu"])
        * world
        * int(cfg["training"].get("gradient_accumulation", 1))
    )
    target_batch = int(cfg["training"].get("target_global_batch_size", effective_batch))
    if effective_batch != target_batch:
        raise ValueError(
            f"Effective global batch is {effective_batch}, but target_global_batch_size={target_batch}. "
            "Adjust batch_size_per_gpu or gradient_accumulation and record the change."
        )
    out_dir = Path(cfg["experiment"]["output_dir"])
    record_budget = bool(cfg["training"].get("record_actual_budget", False))
    if record_budget and world != 1:
        raise RuntimeError("Actual-budget protocol requires one process/GPU per model job")
    if record_budget and not resume and out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError("Refusing to overwrite a nonempty independent training run")
    degrader = _degrader(cfg["degradation"])
    data_cfg = cfg["data"]
    train_set = BraTSPatchDataset(
        data_cfg["train_manifest"], data_cfg["modality"], degrader,
        tuple(data_cfg["patch_size_dhw"]), data_cfg["train_samples_per_epoch"],
        data_cfg.get("foreground_probability", 0.8), data_cfg.get("clip_z", 5.0),
        data_cfg.get("augmentation", {}),
    )
    val_set = BraTSFixedPatchDataset(
        data_cfg["val_manifest"], data_cfg["modality"], degrader,
        tuple(data_cfg["patch_size_dhw"]), data_cfg.get("clip_z", 5.0),
    )
    sampler = DistributedSampler(train_set, shuffle=True, seed=seed) if distributed else None
    train_loader = DataLoader(
        train_set, batch_size=cfg["training"]["batch_size_per_gpu"], shuffle=sampler is None,
        sampler=sampler, num_workers=data_cfg.get("num_workers", 4), pin_memory=True,
        persistent_workers=data_cfg.get("num_workers", 4) > 0, worker_init_fn=worker_seed,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    model = build_model(cfg["model"]).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"]
    )
    total_epochs = int(cfg["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["training"].get("amp", True) and device.type == "cuda")
    criterion = CompositeFDMRNetLoss(**cfg["loss"]).to(device)
    manifests = {
        "train": sha256_file(data_cfg["train_manifest"]),
        "val": sha256_file(data_cfg["val_manifest"]),
    }
    if data_cfg.get("test_manifest"):
        manifests["test"] = sha256_file(data_cfg["test_manifest"])
    if data_cfg.get("external_eval_manifest"):
        manifests["external"] = sha256_file(data_cfg["external_eval_manifest"])
    start_epoch, best = 1, -math.inf
    restored_budget = None
    if resume:
        payload = load_checkpoint(resume, model, optimizer, scheduler, scaler, restore_rng=True, map_location=device)
        if payload.get("config_sha256") != cfg.get("_config_sha256"):
            raise RuntimeError("Resume refused: configuration SHA256 differs from the checkpoint")
        if payload.get("manifest_hashes") != manifests:
            raise RuntimeError("Resume refused: one or more split manifests changed")
        start_epoch, best = int(payload["epoch"]) + 1, float(payload["best_metric"])
        restored_budget = payload.get("training_state")
        if record_budget and not restored_budget:
            raise RuntimeError("Resume refused: checkpoint has no executed-budget state")
    ledger = TrainingBudgetLedger(out_dir, cfg, manifests, world, restored_budget) if record_budget else None
    accumulation = int(cfg["training"].get("gradient_accumulation", 1))
    if record_budget and (len(train_loader) % accumulation or len(train_set) % int(cfg["training"]["batch_size_per_gpu"])):
        raise ValueError("Frozen budget requires complete batch/accumulation boundaries")
    if record_budget and data_cfg.get("num_workers", 4) != 0:
        raise ValueError("Executed-budget resume protocol requires num_workers=0; worker RNG is not checkpointed")
    # Do not rewrite configuration/status before resume provenance validation.
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_resolved_config(cfg, out_dir / "config.yaml")
        atomic_json_dump({"status": "starting", "world_size": world}, out_dir / "job_status.json")
    max_epoch = min(total_epochs, stop_after_epoch or total_epochs)
    try:
        for epoch in range(start_epoch, max_epoch + 1):
            if sampler:
                sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            sums: dict[str, float] = {}
            component_steps = 0
            t0 = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for step, batch in enumerate(train_loader, 1):
                batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
                batch["lr"], batch["field_target"] = _geometry_augment(
                    batch["lr"], batch["hr"].shape[-3:], cfg["training"].get("geometry_augmentation", {})
                )
                with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    outputs = model(batch["lr"], tuple(batch["hr"].shape[-3:]))
                    if "pred_low" in outputs:
                        loss, components = criterion(outputs, batch)
                    else:
                        loss = masked_l1(outputs["pred"], batch["hr"], batch.get("brain_mask"))
                        components = {"total": loss, "rec": loss}
                    loss = loss / accumulation
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Non-finite training loss")
                scaler.scale(loss).backward()
                step_result = None
                if step % accumulation == 0 or step == len(train_loader):
                    step_result = complete_amp_optimizer_step(
                        scaler=scaler, optimizer=optimizer, scheduler=None,
                        parameters=model.named_parameters(), gradient_clip=cfg["training"].get("gradient_clip", 1.0))
                    optimizer.zero_grad(set_to_none=True)
                if ledger:
                    ledger.micro_batch(epoch, step, batch["hr"].shape[0], components["total"].detach().item(), step_result)
                for key, value in components.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach().item())
                component_steps += 1
            scheduler.step()
            if ledger:
                ledger.epoch_completed(epoch)
            if distributed:
                keys = sorted(sums)
                packed = torch.tensor(
                    [*(sums[key] for key in keys), float(component_steps)],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                sums = {key: float(packed[i].item()) for i, key in enumerate(keys)}
                component_steps = int(packed[-1].item())
            if distributed:
                dist.barrier()
            if rank == 0:
                eval_model = model.module if hasattr(model, "module") else model
                eval_model.eval()
                val_values = []
                with torch.inference_mode():
                    for batch in val_loader:
                        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                            pred = eval_model(batch["lr"], tuple(batch["hr"].shape[-3:]))["pred"]
                        val_values.append(metric_bundle(pred.float(), batch["hr"].float(), batch["brain_mask"], 1.0))
                val_psnr = sum(v["psnr"] for v in val_values) / len(val_values)
                val_ssim = sum(v["ssim"] for v in val_values) / len(val_values)
                elapsed = time.perf_counter() - t0
                row = {
                    "epoch": epoch,
                    **{f"train_{k}": value / max(component_steps, 1) for k, value in sums.items()},
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch_seconds": elapsed,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0,
                }
                append_csv(out_dir / "epoch_metrics.csv", row)
                print(json.dumps(row, sort_keys=True), flush=True)
                improved = val_psnr > best
                best = max(best, val_psnr)
                common = dict(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch,
                              best_metric=best, config=cfg, manifest_hashes=manifests,
                              training_state=ledger.state() if ledger else None)
                save_checkpoint(out_dir / "latest.pt", **common)
                if ledger:
                    ledger.bind_checkpoint(out_dir / "latest.pt")
                if improved:
                    save_checkpoint(out_dir / "best.pt", **common)
                    if ledger:
                        ledger.bind_checkpoint(out_dir / "best.pt")
                if epoch in set(cfg["training"].get("checkpoint_epochs", [100, 300])) or epoch == max_epoch:
                    save_checkpoint(out_dir / f"epoch_{epoch:03d}.pt", **common)
                    if ledger:
                        ledger.bind_checkpoint(out_dir / f"epoch_{epoch:03d}.pt")
                atomic_json_dump({"status": "running", "completed_epoch": epoch, "best_val_psnr": best}, out_dir / "job_status.json")
            if distributed:
                dist.barrier()
        if rank == 0:
            state = "stopped_for_manual_review" if max_epoch < total_epochs else "training_completed"
            atomic_json_dump({"status": state, "completed_epoch": max_epoch, "best_val_psnr": best}, out_dir / "job_status.json")
    except Exception as exc:
        if rank == 0:
            atomic_json_dump(
                {"status": "safe_stopped_error", "error": repr(exc), "traceback": traceback.format_exc()},
                out_dir / "job_status.json",
            )
        raise
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
