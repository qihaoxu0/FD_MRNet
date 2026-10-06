#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from torch.utils.data import DataLoader

from fdmrnet.config import load_config
from fdmrnet.data import BraTSPatchDataset
from fdmrnet.degradation import ThroughPlaneDegrader
from fdmrnet.utils.reproducibility import seed_everything, worker_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--patch", nargs=3, type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["experiment"]["seed"]), cfg["training"].get("deterministic", True))
    degrader = ThroughPlaneDegrader(**{key: value for key, value in cfg["degradation"].items() if key != "axis"})
    dataset = BraTSPatchDataset(
        cfg["data"]["train_manifest"], cfg["data"]["modality"], degrader, tuple(args.patch),
        args.warmup + args.batches, cfg["data"].get("foreground_probability", 0.8),
        cfg["data"].get("clip_z", 5.0), cfg["data"].get("augmentation", {}),
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        pin_memory=args.pin_memory, persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_seed,
    )
    intervals, previous = [], time.perf_counter()
    shapes = None
    for index, batch in enumerate(loader):
        now = time.perf_counter()
        if index >= args.warmup:
            intervals.append(now - previous)
        previous = now
        shapes = {"lr": list(batch["lr"].shape), "hr": list(batch["hr"].shape)}
    report = {
        "num_workers": args.num_workers, "pin_memory": args.pin_memory,
        "warmup_batches": args.warmup, "measured_batches": args.batches,
        "mean_batch_seconds": statistics.mean(intervals),
        "std_batch_seconds": statistics.stdev(intervals) if len(intervals) > 1 else 0.0,
        "batches_per_second": 1.0 / statistics.mean(intervals), "shapes": shapes,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
