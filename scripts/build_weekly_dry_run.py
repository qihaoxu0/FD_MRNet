#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from fdmrnet.config import load_config, sha256_file


def clean(cfg):
    return {key: value for key, value in cfg.items() if not key.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve(); root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    base = clean(load_config(repo / "configs/weekly/frozen_3060ti_7day.yaml"))
    registry = yaml.safe_load((repo / "EXPERIMENT_REGISTRY.yaml").read_text(encoding="utf-8"))
    tasks = []

    def add(priority, dataset_modality, scale, method, seed=2025, ablation=None, policy="normal", requirement_class="required"):
        modality = dataset_modality.lower()
        run_id = f"p{priority}_brats2021_{modality}_x{scale}_{method}_seed{seed}"
        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg["experiment"] = {"id": run_id, "seed": seed, "output_dir": str((root.parent / "weekly_outputs" / run_id).as_posix())}
        cfg["data"]["modality"] = modality
        cfg["degradation"].update({"scale": scale, "sigma_vox": 0.8493 if scale == 2 else 1.6985})
        if ablation:
            ab_cfg = clean(load_config(repo / f"configs/ablations/{ablation}.yaml"))
            cfg["model"] = ab_cfg["model"]
            cfg["loss"] = ab_cfg["loss"]
        else:
            model_key = method if method in registry["methods"] else "fdmrnet"
            cfg["model"] = dict(registry["methods"][model_key]["model"])
        cfg["model"]["gradient_checkpointing"] = True
        config_path = root / "configs" / f"{run_id}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        tasks.append({
            "queue_index": len(tasks) + 1, "priority": priority, "run_id": run_id,
            "modality": modality, "scale": scale, "method": method, "seed": seed,
            "ablation": ablation or "", "launch_policy": policy, "status": "pending",
            "requirement_class": requirement_class,
            "config": str(config_path), "output_dir": cfg["experiment"]["output_dir"],
            "maximum_micro_steps": 6000, "minimum_micro_steps": 3000,
            "estimated_step_seconds_conservative": 3.02504,
            "estimated_training_hours": round(6000 * 3.02504 / 3600, 3),
            "estimated_validation_checkpoint_hours": 0.727,
            "estimated_total_hours": round(6000 * 3.02504 / 3600 + 0.727, 3),
            "config_sha256": sha256_file(config_path),
        })

    for method, modality in (
        ("fdmrnet", "t1"), ("fdmrnet", "t2"),
        ("swinir3d", "t1"), ("swinir3d", "t2"),
        ("matched_residual3d", "t1"), ("matched_residual3d", "t2"),
    ):
        add(1, modality, 4, method)
    for modality in ("t1", "t2"):
        add(2, modality, 2, "fdmrnet")
    for method, seed in (("fdmrnet", 2026), ("fdmrnet", 2027), ("swinir3d", 2026), ("swinir3d", 2027)):
        add(3, "t1", 4, method, seed)
    module_ablations = [
        ("without_frequency", "no_frequency", "normal"),
        ("single_deformation_field", "single_field", "normal"),
        ("without_cross_frequency_gate", "no_cross_gate", "normal"),
        ("without_localized_attention", "no_attention", "normal"),
        ("without_adaptive_fusion", "fixed_fusion", "normal"),
        ("without_global_residual", "no_global_residual", "normal"),
    ]
    for method, ablation, policy in module_ablations:
        add(4, "t1", 4, method, ablation=ablation, policy=policy)
    add(5, "t1", 4, "without_frequency_loss", ablation="no_frequency_loss")
    add(5, "t1", 4, "without_deformation_supervision", ablation="no_field_supervision")
    add(6, "t1", 4, "without_smoothness_regularization", ablation="no_smoothness_loss",
        policy="deferred", requirement_class="opportunistic")
    add(6, "t1", 4, "without_auxiliary_reconstruction_loss", ablation=None,
        policy="deferred_protocol_definition", requirement_class="opportunistic")
    queue = root.parent / "weekly_run_queue.csv"
    with queue.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tasks[0])); writer.writeheader(); writer.writerows(tasks)
    core = [task for task in tasks if task["requirement_class"] == "required"]
    opportunistic = [task for task in tasks if task["requirement_class"] == "opportunistic"]
    core_hours = sum(t["estimated_total_hours"] for t in core)
    optional_hours = sum(t["estimated_total_hours"] for t in opportunistic)
    required_weekly_hours = 89.2895 / 60 + core_hours + 36
    report = {
        "total_rows": len(tasks), "required_tasks": len(core), "opportunistic_tasks": len(opportunistic),
        "priority_counts": {str(p): sum(t["priority"] == p for t in tasks) for p in range(1, 7)},
        "expected_active": len(core),
        "expected_deferred": len(opportunistic),
        "selection_elapsed_hours": 89.2895 / 60,
        "core_training_estimated_hours_after_selection": core_hours,
        "training_total_from_week_start_hours": 89.2895 / 60 + core_hours,
        "opportunistic_additional_hours": optional_hours,
        "evaluation_reserved_hours": 36,
        "weekly_total_estimated_hours": required_weekly_hours,
        "weekly_total_if_opportunistic_added_hours": required_weekly_hours + optional_hours,
        "opportunistic_decision": "deferred because adding tasks 21-22 exceeds the 162-hour target",
        "disk_estimate_gib": round(len(core) * 1.1 + 20, 1),
        "notes": [
            "Conservative task timing uses FD-MRNet step time for every method; baselines may be faster.",
            "All 20 required tasks remain intact; no required experiment has reduced steps or cases.",
            "The auxiliary-reconstruction-loss mapping is scientifically ambiguous and is blocked rather than guessed.",
        ],
    }
    (root.parent / "weekly_dry_run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
