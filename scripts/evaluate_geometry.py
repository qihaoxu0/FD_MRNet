#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fdmrnet.config import load_config
from fdmrnet.data import BraTSPatchDataset
from fdmrnet.degradation import ThroughPlaneDegrader
from fdmrnet.engine.checkpoint import load_checkpoint
from fdmrnet.geometry import invert_displacement, resize_displacement, smooth_random_field, warp_volume
from fdmrnet.metrics.geometry import geometry_metrics
from fdmrnet.models import build_model
from fdmrnet.utils.reproducibility import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", type=int, default=50)
    parser.add_argument("--out", required=True)
    parser.add_argument("--save-examples", type=int, default=3)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["experiment"]["seed"]), cfg["training"].get("deterministic", True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    degrader = ThroughPlaneDegrader(**{k: v for k, v in cfg["degradation"].items() if k != "axis"})
    ds = BraTSPatchDataset(cfg["data"]["test_manifest"], cfg["data"]["modality"], degrader,
                           tuple(cfg["data"]["patch_size_dhw"]), args.cases, 1.0, cfg["data"].get("clip_z", 5.0))
    model = build_model(cfg["model"]).to(device).eval()
    payload = load_checkpoint(args.checkpoint, model, restore_rng=False, map_location=device)
    if payload.get("config_sha256") != cfg.get("_config_sha256"):
        raise RuntimeError("Geometry evaluation refused: checkpoint/config hash mismatch")
    rows = []
    geometry_cfg = cfg["training"].get("geometry_augmentation", {})
    for index, batch in enumerate(DataLoader(ds, batch_size=1, num_workers=0)):
        lr = batch["lr"].to(device)
        target = smooth_random_field(
            1,
            tuple(batch["hr"].shape[-3:]),
            geometry_cfg.get("max_local_displacement_hr_vox", 1.5),
            tuple(geometry_cfg.get("coarse_grid_dhw", [4, 8, 8])),
            geometry_cfg.get("max_translation_hr_vox", 0.5),
            device=device,
            dtype=lr.dtype,
        )
        field_lr = resize_displacement(target, tuple(lr.shape[-3:]))
        corrupted = warp_volume(
            lr, invert_displacement(field_lr, geometry_cfg.get("inverse_iterations", 7))
        )
        with torch.inference_mode():
            pred = model(corrupted, tuple(batch["hr"].shape[-3:]))["d_composite"]
        rows.append({"case": index, "subject": batch["subject"][0], **geometry_metrics(pred, target)})
        if index < args.save_examples:
            example_dir = Path(args.out).parent / "geometry_examples"
            example_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                example_dir / f"case_{index:03d}.npz",
                subject=batch["subject"][0],
                predicted_field=pred[0].float().cpu().numpy(),
                target_field=target[0].float().cpu().numpy(),
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
