#!/usr/bin/env python
"""Paired subject statistics with immutable formal sources and diagnostic separation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

from build_tables import load_long, validate_seed_and_subject_sets

CONDITION = ["dataset", "split", "training_regime", "modality", "scale", "metric"]
REQUIRED = set(CONDITION + ["group", "method", "region", "subject", "seed", "value", "evidence_class"])


def bootstrap_ci(values: np.ndarray, seed: int = 2025, draws: int = 10000):
    if len(values) < 2:
        return "N/A"
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return np.quantile(samples, [0.025, 0.975]).astype(float).tolist()


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted.tolist()


def safe_wilcoxon(model: np.ndarray, comparator: np.ndarray):
    if len(model) < 2:
        return "N/A"
    delta = model - comparator
    if np.all(delta == 0):
        return 1.0
    return float(wilcoxon(model, comparator, alternative="greater", zero_method="wilcox").pvalue)


def paired_statistics(long: pd.DataFrame, seed: int = 2025) -> list[dict]:
    missing = REQUIRED - set(long.columns)
    if missing:
        raise RuntimeError(f"Missing statistics metadata: {sorted(missing)}")
    primary = long[(long["group"].isin(["main", "baselines"])) & (long.region == "brain")
                   & (long.metric.isin(["psnr", "ssim"]))].copy()
    if primary.empty:
        return []
    if primary[list(REQUIRED - {"value"})].isna().any().any():
        raise RuntimeError("Missing subject/seed/comparison metadata")
    primary["value"] = pd.to_numeric(primary.value, errors="coerce")
    if not np.isfinite(primary.value).all():
        raise RuntimeError("Primary PSNR/SSIM values must be finite")
    unique = CONDITION + ["method", "subject", "seed"]
    if primary.duplicated(unique).any():
        raise RuntimeError("Duplicate seed-subject-model metrics")
    for key, frame in primary.groupby(CONDITION):
        methods = list(frame.groupby("method"))
        seed_sets = [set(values.seed) for _, values in methods]
        if any(values != seed_sets[0] for values in seed_sets[1:]):
            raise RuntimeError(f"Unpaired seed IDs for {key}")
        rosters = []
        for method, values in methods:
            per_seed = [set(rows.subject) for _, rows in values.groupby("seed")]
            if any(subjects != per_seed[0] for subjects in per_seed[1:]):
                raise RuntimeError(f"Unpaired subjects between seeds for {key}, {method}")
            rosters.append(per_seed[0])
        if any(subjects != rosters[0] for subjects in rosters[1:]):
            raise RuntimeError(f"Unpaired subjects across models for {key}")
    # Each patient enters inference once, after averaging its evaluated seeds.
    subject = primary.groupby(CONDITION + ["method", "subject"], as_index=False)["value"].mean()
    comparisons = []
    for condition, frame in subject.groupby(CONDITION):
        proposed = frame[frame.method == "fdmrnet"][["subject", "value"]].rename(columns={"value": "proposed"})
        if proposed.empty:
            continue
        for method in sorted(set(frame.method) - {"fdmrnet"}):
            other = frame[frame.method == method][["subject", "value"]].rename(columns={"value": "comparator"})
            pair = proposed.merge(other, on="subject", how="outer", validate="one_to_one", indicator=True)
            if not pair["_merge"].eq("both").all():
                raise RuntimeError("Pairing would drop unmatched patients")
            delta = (pair.proposed - pair.comparator).to_numpy(float)
            row = dict(zip(CONDITION, condition))
            row["scale"] = int(row["scale"])
            row.update(comparison=f"fdmrnet > {method}", hypothesis="prespecified one-sided greater",
                       inferential_unit="patients after seed averaging; conditional on evaluated seeds",
                       n_subjects=len(pair), fdmrnet_mean=float(pair.proposed.mean()),
                       comparator_mean=float(pair.comparator.mean()), delta_mean=float(delta.mean()),
                       delta_ci95=bootstrap_ci(delta, seed),
                       p_raw=safe_wilcoxon(pair.proposed.to_numpy(float), pair.comparator.to_numpy(float)),
                       inferential_status="available" if len(pair) >= 2 else "N/A: fewer than two patients")
            comparisons.append(row)
    eligible = [index for index, row in enumerate(comparisons) if isinstance(row["p_raw"], float)]
    adjusted = holm([comparisons[index]["p_raw"] for index in eligible]) if eligible else []
    for row in comparisons:
        row["p_holm"] = "N/A"
        row["holm_family"] = "all inferential comparisons in this output"
    for index, value in zip(eligible, adjusted):
        comparisons[index]["p_holm"] = value
    return comparisons


def require_matching_long(provided: pd.DataFrame, verified: pd.DataFrame) -> None:
    if set(provided.columns) != set(verified.columns):
        raise RuntimeError("Long CSV schema differs from verified registry-derived data")
    order = CONDITION + ["group", "method", "region", "subject", "seed"]
    columns = sorted(verified.columns)
    try:
        pd.testing.assert_frame_equal(provided.sort_values(order).reset_index(drop=True)[columns],
                                      verified.sort_values(order).reset_index(drop=True)[columns],
                                      check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-14)
    except AssertionError as exc:
        raise RuntimeError("Long CSV differs from verified registry-derived data") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-long", help="subject_seed_metrics_long.csv from build_tables.py")
    parser.add_argument("--registry", help="Immutable schema-3 result registry required for formal statistics")
    parser.add_argument("--submission-mode", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    if args.submission_mode:
        if not args.registry:
            raise RuntimeError("Formal statistics require --registry and --submission-mode")
        registry = yaml.safe_load(Path(args.registry).read_text(encoding="utf-8"))
        long = load_long(registry, True)
        if long.empty:
            raise RuntimeError("Formal statistics require completed results")
        validate_seed_and_subject_sets(long, 3)
        if args.metrics_long:
            require_matching_long(pd.read_csv(args.metrics_long, dtype={"subject": str}), long)
    else:
        if not args.metrics_long:
            raise RuntimeError("Diagnostic statistics require --metrics-long")
        long = pd.read_csv(args.metrics_long, dtype={"subject": str})
    comparisons = paired_statistics(long, args.seed)
    usage = "submission" if args.submission_mode else "NOT_FOR_SUBMISSION"
    for row in comparisons:
        row["usage"] = usage
    out = Path(args.out)
    if args.submission_mode and (out / "NOT_FOR_SUBMISSION.txt").exists():
        raise RuntimeError("Use a new directory for formal statistics")
    out.mkdir(parents=True, exist_ok=True)
    if not args.submission_mode:
        (out / "NOT_FOR_SUBMISSION.txt").write_text("Diagnostic statistics only; no manuscript inferential claims.\n", encoding="utf-8")
    (out / "paired_statistics.json").write_text(json.dumps(comparisons, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(comparisons).to_csv(out / "paired_statistics.csv", index=False, na_rep="N/A")


if __name__ == "__main__":
    main()
