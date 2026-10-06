#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from collect_results import verify_result, require_hash


FORBIDDEN = ("SIMULATED", "PLACEHOLDER", "NOT_FOR_SUBMISSION", "NOT FOR SUBMISSION")
METRICS = ("psnr", "ssim", "nmse", "hfen")
REGIONS = ("brain", "tumor")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_ci(values: np.ndarray, seed: int = 2025, draws: int = 10000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def load_long(registry: dict, submission_mode: bool) -> pd.DataFrame:
    if submission_mode:
        if registry.get("schema_version") != 3 or registry.get("missing_runs"):
            raise RuntimeError("Submission mode requires complete provenance registry schema 3")
        if not registry.get("source_run_manifest") or not registry.get("source_run_manifest_sha256"):
            raise RuntimeError("Submission mode requires the frozen source run manifest")
        require_hash(Path(registry["source_run_manifest"]), registry["source_run_manifest_sha256"], "run manifest")
    rows: list[dict] = []
    seen_ids: set[str] = set()
    baseline_conditions: set[tuple] = set()
    for item in registry.get("completed_results", []):
        if submission_mode and item.get("training_budget_sha256", "unknown_diagnostic") == "unknown_diagnostic":
            raise RuntimeError("Formal registry lacks a verified executed training budget")
        if item["id"] in seen_ids:
            raise RuntimeError(f"Duplicate completed result id: {item['id']}")
        seen_ids.add(item["id"])
        path = Path(item["metrics_csv"])
        frame, audit = verify_result(item, submission_mode)
        metadata = {
            "group": item.get("group", "main"),
            "dataset": str(item["dataset"]),
            "modality": str(item["modality"]).upper(),
            "scale": int(item["scale"]),
            "method": str(item["method"]),
            "seed": int(item["seed"]),
            "training_regime": item.get("training_regime", "shared"),
            "split": str(item.get("split", "test")),
            "protocol_sha256": item.get("protocol_sha256", "unbound_diagnostic"),
            "comparison_policy_sha256": item.get("comparison_policy_sha256", "unbound_diagnostic"),
            "training_budget_sha256": item.get("training_budget_sha256", "unknown_diagnostic"),
            "split_sha256": item.get("split_sha256", "unbound_diagnostic"),
            "evidence_class": item.get("evidence_class", "diagnostic"),
            "segmentation_available": item.get("segmentation_available", True),
        }
        prefixes = [(str(item.get("model_prefix", "sr")), metadata["method"])]
        condition = (metadata["group"], metadata["dataset"], metadata["split"], metadata["modality"], metadata["scale"], metadata["seed"], metadata["training_regime"])
        if metadata["method"] == "fdmrnet" and condition not in baseline_conditions:
            prefixes.append((str(item.get("baseline_prefix", "baseline")), "trilinear"))
            baseline_conditions.add(condition)
        for prefix, method in prefixes:
            for region in REGIONS:
                for metric in METRICS:
                    column = f"{prefix}_{region}_{metric}"
                    if column not in frame:
                        raise RuntimeError(f"Missing {column} in {path}")
                    values = frame[column]
                    for subject, value in zip(frame["subject"].astype(str), values):
                        rows.append(metadata | {"method": method, "subject": subject, "region": region, "metric": metric, "value": value})
    return pd.DataFrame(rows)


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    keys = ["group", "dataset", "split", "modality", "scale", "method", "training_regime", "region", "metric"]
    if long.empty:
        return pd.DataFrame()
    # Average repeated seeds per subject first; subjects, not patches or seed-subject
    # duplicates, are the inferential units.
    subject = long.groupby(keys + ["subject"], as_index=False, dropna=False)["value"].mean()
    seed_means = long.groupby(keys + ["seed"], as_index=False, dropna=False)["value"].mean()
    output = []
    for group_key, frame in subject.groupby(keys, dropna=False):
        finite = frame["value"].to_numpy(float)
        finite = finite[np.isfinite(finite)]
        seed_frame = seed_means
        for key, value in zip(keys, group_key):
            seed_frame = seed_frame[seed_frame[key] == value]
        ci_low, ci_high = bootstrap_ci(finite)
        row = dict(zip(keys, group_key))
        row.update(
            {
                "subject_mean": float(np.mean(finite)) if len(finite) else np.nan,
                "subject_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "valid_subjects": int(len(finite)),
                "seed_mean_sd": float(seed_frame["value"].std(ddof=1)),
                "seeds": int(seed_frame["seed"].nunique()),
                "status": "available" if len(finite) else "N/A: no valid tumor ROI or segmentation unavailable",
                "ci_unit": "subjects after seed averaging; conditional on evaluated seeds",
            }
        )
        output.append(row)
    return pd.DataFrame(output)


def compact_table(summary: pd.DataFrame, dataset: str, group_names: set[str]) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    subset = summary[
        (summary.dataset.str.lower() == dataset.lower())
        & (summary.region == "brain")
        & (summary.group.isin(group_names))
        & (summary.metric.isin(METRICS))
    ].copy()
    if subset.empty:
        return pd.DataFrame()
    if subset.duplicated(["method", "modality", "split", "training_regime", "scale", "metric"]).any():
        raise RuntimeError("Ambiguous duplicate method condition across result groups; no table value was selected")
    subset["formatted"] = subset.apply(
        lambda row: (f"{row.subject_mean:.4f} +/- {row.subject_sd:.4f}" if np.isfinite(row.subject_sd) else f"{row.subject_mean:.4f} (SD N/A)") if np.isfinite(row.subject_mean) else "N/A", axis=1
    )
    pivot = subset.pivot_table(
        index=["method", "modality", "split", "training_regime"], columns=["scale", "metric"], values="formatted", aggfunc="first"
    )
    pivot.columns = [f"x{scale}_{metric.upper()}" for scale, metric in pivot.columns]
    return pivot.reset_index().sort_values(["modality", "method"])


def write_markdown_table(frame: pd.DataFrame, path: Path, title: str) -> None:
    note = (
        "Values are subject-level full-volume mean +/- across-subject SD after averaging seed-specific "
        "scores for each subject. Percentile bootstrap 95% CIs and valid n are in main_results_long.csv. "
        "These CIs describe subject sampling conditional on the evaluated seeds; seed_mean_sd is separate."
    )
    path.write_text(f"# {title}\n\n{frame.to_markdown(index=False)}\n\n{note}\n", encoding="utf-8")


def validate_seed_and_subject_sets(long: pd.DataFrame, expected_seeds: int) -> None:
    if long.empty:
        return
    condition = ["group", "dataset", "split", "modality", "scale", "method", "training_regime"]
    unique = condition + ["seed", "subject", "region", "metric"]
    if long.duplicated(unique).any():
        raise RuntimeError("Duplicate seed-subject metrics within an evaluation condition")
    seed_counts = long.groupby(condition)["seed"].nunique()
    bad = seed_counts[seed_counts != expected_seeds] if expected_seeds else seed_counts.iloc[:0]
    if len(bad):
        raise RuntimeError(f"Conditions must have exactly {expected_seeds} seeds:\n{bad}")
    for key, frame in long.groupby(condition + ["region", "metric"]):
        sets = [set(values.subject) for _, values in frame.groupby("seed")]
        valid_sets = [set(values.loc[values.value.notna(), "subject"]) for _, values in frame.groupby("seed")]
        if any(subjects != sets[0] for subjects in sets[1:]):
            raise RuntimeError(f"Subject set mismatch between seeds for {key}")
        if any(subjects != valid_sets[0] for subjects in valid_sets[1:]):
            raise RuntimeError(f"Valid subject set mismatch between seeds for {key}")
    brain_psnr = long[
        (long.region == "brain")
        & (long.metric == "psnr")
    ]
    for key, frame in brain_psnr.groupby(["dataset", "split", "modality", "scale", "training_regime"]):
        sets = [set(values.subject) for _, values in frame.groupby(["group", "method"])]
        if sets and any(subjects != sets[0] for subjects in sets[1:]):
            raise RuntimeError(f"Subject set mismatch within comparison condition {key}")
        seeds = [set(values.seed) for _, values in frame.groupby(["group", "method"])]
        if any(values != seeds[0] for values in seeds[1:]):
            raise RuntimeError(f"Seed IDs mismatch within comparison condition {key}")
        for field in ("comparison_policy_sha256", "split_sha256", "training_budget_sha256"):
            if frame[field].nunique() > 1:
                raise RuntimeError(f"{field} mismatch within comparison condition {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--submission-mode", action="store_true")
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--required-datasets", nargs="+", default=["BraTS2021", "BraTS2023"],
                        help="Datasets required in submission mode; specify one for an independent external table.")
    args = parser.parse_args()
    if args.submission_mode and args.expected_seeds != 3:
        raise RuntimeError("Formal revision requires exactly three declared seeds")
    registry = yaml.safe_load(Path(args.registry).read_text(encoding="utf-8"))
    long = load_long(registry, args.submission_mode)
    if args.submission_mode and long.empty:
        raise RuntimeError("Submission mode requires real completed results")
    validate_seed_and_subject_sets(long, args.expected_seeds if args.submission_mode else 0)
    summary = summarize(long)
    if args.submission_mode:
        for dataset in args.required_datasets:
            if compact_table(summary, dataset, {"main", "baselines"}).empty:
                raise RuntimeError(f"Missing formal results for {dataset}; no artifacts were written")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.submission_mode:
        (out / "NOT_FOR_SUBMISSION.txt").write_text("Technical diagnostic output; not validated submission evidence.\n", encoding="utf-8")
    elif (out / "NOT_FOR_SUBMISSION.txt").exists():
        raise RuntimeError("Use a new output directory; this directory is marked NOT_FOR_SUBMISSION")
    long.to_csv(out / "subject_seed_metrics_long.csv", index=False, na_rep="N/A")
    summary.to_csv(out / "main_results_long.csv", index=False, na_rep="N/A")
    if not summary.empty:
        summary.to_latex(out / "main_results_long.tex", index=False, float_format="%.6f", na_rep="N/A")
    for number, dataset in (("I", "BraTS2021"), ("II", "BraTS2023")):
        table = compact_table(summary, dataset, {"main", "baselines"})
        if args.submission_mode and dataset.lower() in {name.lower() for name in args.required_datasets} and table.empty:
            raise RuntimeError(f"Missing formal results for {dataset}; Table {number} cannot be built")
        if not table.empty:
            table.to_csv(out / f"Table_{number}_{dataset}.csv", index=False, na_rep="N/A")
            table.to_latex(out / f"Table_{number}_{dataset}.tex", index=False, escape=False, na_rep="N/A")
            title = f"Table {number}: {dataset} through-plane SR"
            write_markdown_table(table, out / f"Table_{number}_{dataset}.md", title if args.submission_mode else "NOT_FOR_SUBMISSION: " + title)
    ablation_source = summary[
        (summary.dataset.str.lower() == "brats2021")
        & (summary.modality.str.upper() == "T1")
        & (summary.scale == 4)
    ] if not summary.empty else summary
    ablation = compact_table(ablation_source, "BraTS2021", {"main", "ablations"})
    if not ablation.empty:
        ablation.to_csv(out / "Table_III_Ablations.csv", index=False, na_rep="N/A")
        ablation.to_latex(out / "Table_III_Ablations.tex", index=False, escape=False, na_rep="N/A")


if __name__ == "__main__":
    main()
