from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fdmrnet.config import sha256_file
from fdmrnet.data import BraTSFullVolumeDataset
from fdmrnet.degradation import ThroughPlaneDegrader
from fdmrnet.engine.checkpoint import load_checkpoint
from fdmrnet.engine.sliding_window import sliding_window_predict
from fdmrnet.metrics import metric_bundle
from fdmrnet.models import build_model
from fdmrnet.utils.io import atomic_json_dump


def evaluate_full_volume(
    cfg: dict,
    checkpoint: str,
    split: str,
    tag: str | None = None,
    save_subjects: set[str] | None = None,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dcfg = {k: v for k, v in cfg["degradation"].items() if k != "axis"}
    degrader = ThroughPlaneDegrader(**dcfg)
    external_without_seg = split == "external"
    manifest_key = "external_eval_manifest" if external_without_seg else f"{split}_manifest"
    manifest = cfg["data"][manifest_key]
    if external_without_seg and not cfg["data"].get("external_eval_without_seg", False):
        raise RuntimeError("External evaluation without segmentation requires explicit config opt-in")
    dataset = BraTSFullVolumeDataset(
        manifest, cfg["data"]["modality"], degrader, cfg["data"].get("clip_z", 5.0),
        external_eval_without_seg=external_without_seg,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = build_model(cfg["model"]).to(device)
    payload = load_checkpoint(checkpoint, model, restore_rng=False, map_location=device)
    if payload.get("config_sha256") != cfg.get("_config_sha256"):
        raise RuntimeError("Evaluation refused: checkpoint and configuration hashes differ")
    if payload.get("manifest_hashes", {}).get(split) != sha256_file(manifest):
        raise RuntimeError(f"Evaluation refused: {split} manifest changed after training")
    model.eval()
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or f"{split}_epoch{Path(checkpoint).stem.split('_')[-1]}"
    csv_path = output_dir / f"{tag}_subject_metrics.csv"
    rows, audits = [], []
    patch = tuple(cfg["evaluation"]["patch_size_dhw"])
    overlap = tuple(cfg["evaluation"]["overlap_dhw"])
    data_range = float(cfg["evaluation"].get("data_range", 1.0))
    model_name = str(cfg["model"]["name"]).lower()
    save_subjects = save_subjects or set(cfg["evaluation"].get("visualization_subjects", []))
    prediction_dir = output_dir / f"{tag}_predictions"
    for batch in loader:
        subject = batch["subject"][0]
        lr, target = batch["lr"].to(device), batch["hr"].to(device)
        brain, tumor = batch["brain_mask"].to(device), batch["tumor_mask"].to(device)
        coarse = degrader.coarse(lr, tuple(target.shape[-3:]))
        torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
        t0 = time.perf_counter()
        pred, window_audit = sliding_window_predict(
            model, coarse, patch, overlap, cfg["evaluation"].get("gaussian_sigma_scale", 0.125), cfg["evaluation"].get("amp", True)
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        original_depth = int(batch["original_depth"].item())
        pred, coarse, target = pred[:, :, :original_depth], coarse[:, :, :original_depth], target[:, :, :original_depth]
        brain, tumor = brain[:, :, :original_depth], tumor[:, :, :original_depth]
        baseline_brain = metric_bundle(coarse, target, brain, data_range)
        model_brain = metric_bundle(pred, target, brain, data_range)
        baseline_tumor = None if external_without_seg else metric_bundle(coarse, target, tumor, data_range)
        model_tumor = None if external_without_seg else metric_bundle(pred, target, tumor, data_range)
        row = {
            "subject": subject,
            "dataset": cfg["data"]["dataset"],
            "modality": cfg["data"]["modality"],
            "scale": int(cfg["degradation"]["scale"]),
            "method": model_name,
            "seed": int(cfg["experiment"]["seed"]),
            "seconds": seconds,
        }
        metric_groups = {"baseline_brain": baseline_brain, "sr_brain": model_brain}
        if external_without_seg:
            row["tumor_metrics"] = "N/A: segmentation unavailable"
        else:
            metric_groups.update({"baseline_tumor": baseline_tumor, "sr_tumor": model_tumor})
        for prefix, values in metric_groups.items():
            row.update({f"{prefix}_{key}": value for key, value in values.items()})
        rows.append(row)
        if subject in save_subjects:
            prediction_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                prediction_dir / f"{subject}.npz",
                subject=subject,
                coarse=coarse[0, 0].float().cpu().numpy(),
                prediction=pred[0, 0].float().cpu().numpy(),
                target=target[0, 0].float().cpu().numpy(),
                brain_mask=brain[0, 0].cpu().numpy().astype(np.uint8),
                **({} if external_without_seg else {"tumor_mask": tumor[0, 0].cpu().numpy().astype(np.uint8)}),
            )
        audits.append({
            "subject": subject,
            "target_shape_dhw": list(target.shape[-3:]),
            "lr_shape_dhw": list(lr.shape[-3:]),
            "seconds": seconds,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0,
            **window_audit,
        })
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    audit = {
        "status": "completed",
        "split": split,
        "cases": len(rows),
        "manifest_sha256": sha256_file(manifest),
        "checkpoint_sha256": sha256_file(checkpoint),
        "metric_protocol": {
            "unit": "subject-level full 3D volume",
            "data_range": data_range,
            "mask": (
                "HR nonzero foreground; tumor metrics N/A: segmentation unavailable"
                if external_without_seg else "HR nonzero foreground; tumor label > 0"
            ),
            "ssim_window": 7,
            "ssim_sigma": 1.5,
            "ssim_dimensionality": "3D Gaussian window; subject metric computed before cohort aggregation",
        },
        "dataset": cfg["data"]["dataset"],
        "modality": cfg["data"]["modality"],
        "scale": int(cfg["degradation"]["scale"]),
        "method": model_name,
        "seed": int(cfg["experiment"]["seed"]),
        "degradation": cfg["degradation"],
        "normalization": {"scope": "per HR subject nonzero voxels", "clip_z": cfg["data"].get("clip_z", 5.0)},
        "saved_prediction_subjects": sorted(save_subjects),
        "external_evaluation_policy": (
            "BraTS2023 GLI validation non-overlapping external evaluation set; never used for training, tuning, or checkpoint selection"
            if external_without_seg else None
        ),
        "cases_audit": audits,
    }
    atomic_json_dump(audit, output_dir / f"{tag}_audit.json")
    return csv_path
