#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import shlex
from pathlib import Path

import yaml

from fdmrnet.config import load_config


def clean_config(cfg: dict) -> dict:
    return {key: value for key, value in cfg.items() if not key.startswith("_")}


def add_run(rows: list[dict], cfg: dict, group: str, method: str, seed: int, root: Path) -> None:
    dataset = str(cfg["data"]["dataset"]).lower()
    modality = str(cfg["data"]["modality"]).lower()
    scale = int(cfg["degradation"]["scale"])
    run_id = f"{dataset}_{modality}_x{scale}_{method}_seed{seed}"
    cfg["experiment"] = {
        "id": run_id,
        "seed": int(seed),
        "output_dir": f"outputs/formal/{dataset}_{modality}_x{scale}/{method}/seed{seed}",
    }
    path = root / group / f"{run_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    rows.append(
        {
            "run_id": run_id,
            "group": group,
            "dataset": cfg["data"]["dataset"],
            "modality": modality,
            "scale": scale,
            "method": method,
            "seed": seed,
            "config": str(path.as_posix()),
            "output_dir": cfg["experiment"]["output_dir"],
            "train_command": "torchrun --standalone --nproc_per_node=4 scripts/train.py --config "
            + shlex.quote(str(path.as_posix())),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="EXPERIMENT_REGISTRY.yaml")
    parser.add_argument("--out-dir", default="generated_configs")
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("main", "baselines", "ablations"),
        default=["main", "baselines", "ablations"],
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    registry = yaml.safe_load(Path(args.registry).read_text(encoding="utf-8"))
    seeds = args.seeds or [int(value) for value in registry["seeds"]]
    out_root = Path(args.out_dir)
    rows: list[dict] = []

    if "main" in args.groups:
        method_cfg = registry["methods"]["fdmrnet"]["model"]
        for config_path in registry["main_configs"]:
            for seed in seeds:
                cfg = clean_config(load_config(config_path))
                cfg["model"] = dict(method_cfg)
                add_run(rows, cfg, "main", "fdmrnet", seed, out_root)

    if "baselines" in args.groups:
        for config_path in registry["main_configs"]:
            for method in registry["baseline_methods"]:
                for seed in seeds:
                    cfg = clean_config(load_config(config_path))
                    cfg["model"] = dict(registry["methods"][method]["model"])
                    add_run(rows, cfg, "baselines", method, seed, out_root)

    if "ablations" in args.groups:
        for config_path in registry["ablation_configs"]:
            for seed in seeds:
                cfg = clean_config(load_config(config_path))
                method = str(cfg["experiment"]["id"])
                add_run(rows, cfg, "ablations", method, seed, out_root)

    manifest = out_root / "run_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_root / "all_configs.txt").write_text(
        "\n".join(row["config"] for row in rows) + "\n", encoding="utf-8"
    )
    print(f"Materialized {len(rows)} immutable run configs; manifest={manifest}")


if __name__ == "__main__":
    main()
