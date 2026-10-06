#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from fdmrnet.config import sha256_file
from fdmrnet.data.discovery import discover_brats_subjects, write_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--corrections-root")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--train", type=int)
    parser.add_argument("--val", type=int)
    parser.add_argument("--test", type=int)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()
    records = discover_brats_subjects(args.root, corrections_root=args.corrections_root)
    exact = (args.train, args.val, args.test)
    if any(value is not None for value in exact) and not all(value is not None for value in exact):
        raise ValueError("Provide all of --train/--val/--test together, or omit all three")
    if all(value is not None for value in exact):
        train_n, val_n, test_n = (int(value) for value in exact)
    else:
        if not 0 < args.train_fraction < 1 or not 0 < args.val_fraction < 1:
            raise ValueError("Split fractions must lie in (0,1)")
        train_n = int(len(records) * args.train_fraction)
        val_n = int(len(records) * args.val_fraction)
        test_n = len(records) - train_n - val_n
    if train_n + val_n + test_n != len(records) or min(train_n, val_n, test_n) <= 0:
        raise ValueError(f"Requested {train_n + val_n + test_n} subjects, discovered {len(records)}")
    rng = random.Random(args.seed)
    rng.shuffle(records)
    groups = {
        "train": records[:train_n],
        "val": records[train_n : train_n + val_n],
        "test": records[train_n + val_n :],
    }
    ids = {name: {r.subject for r in rows} for name, rows in groups.items()}
    if ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"]:
        raise RuntimeError("Subject leakage detected")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(sorted(records, key=lambda record: record.subject), out / "all.csv")
    for name, rows in groups.items():
        write_manifest(rows, out / f"{name}.csv")
    audit = {
        "seed": args.seed,
        "root": str(Path(args.root).resolve()),
        "corrections_root": str(Path(args.corrections_root).resolve()) if args.corrections_root else None,
        "counts": {name: len(rows) for name, rows in groups.items()},
        "subjects": {name: sorted(ids[name]) for name in groups},
        "sha256": {name: sha256_file(out / f"{name}.csv") for name in groups},
        "all_manifest_sha256": sha256_file(out / "all.csv"),
        "subject_level_split": True,
        "overlap": 0,
    }
    (out / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"counts": audit["counts"], "sha256": audit["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
