#!/usr/bin/env python
"""Collect explicitly bound evaluation artifacts without choosing on test scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fdmrnet.config import canonical_hash, load_config, sha256_file

METRICS = ("psnr", "ssim", "nmse", "hfen")
FORBIDDEN = ("SIMULATED", "PLACEHOLDER", "NOT_FOR_SUBMISSION", "NOT FOR SUBMISSION", "FIXTURE")
META = ("dataset", "split", "modality", "scale", "method", "seed")


def present(value) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value).strip().lower() in {"true", "1"}:
        return True
    if str(value).strip().lower() in {"false", "0"}:
        return False
    raise RuntimeError(f"Expected boolean segmentation flag, got {value!r}")


def resolve_path(value, base: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute() or path.exists():
        return path.resolve()
    return (base / path).resolve()


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if not expected or sha256_file(path) != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: {path}")


def roster(path: Path) -> list[str]:
    frame = pd.read_csv(path, dtype={"subject": str})
    if "subject" not in frame or frame.empty or frame.subject.isna().any():
        raise RuntimeError(f"Missing/nonempty subject roster: {path}")
    subjects = frame.subject.astype(str).tolist()
    if any(not subject.strip() for subject in subjects) or len(set(subjects)) != len(subjects):
        raise RuntimeError(f"Duplicate or empty subjects in {path}")
    return subjects


def numeric_metric(series: pd.Series, column: str, allow_missing: bool) -> pd.Series:
    missing = series.map(lambda value: pd.isna(value) or str(value).strip().upper().startswith("N/A"))
    values = pd.to_numeric(series.where(~missing), errors="coerce")
    if ((~missing) & (~np.isfinite(values))).any():
        raise RuntimeError(f"Non-numeric or infinite metric in {column}")
    if not allow_missing and missing.any():
        raise RuntimeError(f"Missing required brain metric in {column}")
    finite = values.dropna()
    if column.endswith(("_nmse", "_hfen")) and (finite < 0).any():
        raise RuntimeError(f"Negative error metric in {column}")
    if column.endswith("_ssim") and ((finite < -1) | (finite > 1)).any():
        raise RuntimeError(f"SSIM outside [-1, 1] in {column}")
    return values


def normalized_metadata(value, key: str):
    return int(value) if key in {"scale", "seed"} else str(value).strip().lower()


def comparison_policy(audit: dict, submission_mode: bool = False) -> str:
    """Derive a shared policy hash, excluding model/run-specific protocol bindings."""
    fields = ("metric_protocol", "degradation", "normalization", "inference", "file_identity_audit")
    if any(key not in audit for key in fields):
        if submission_mode:
            raise RuntimeError("Formal audit lacks comparison settings or input identities")
        return "unbound_diagnostic"
    identities = []
    for item in sorted(audit["file_identity_audit"], key=lambda item: item["subject"]):
        identity = {"subject": item["subject"]}
        for role in ("image", "segmentation"):
            if role in item:
                identity[role] = {key: item[role][key] for key in ("original_path", "sha256", "shape_xyz", "affine")}
        if "image" not in identity:
            raise RuntimeError("Missing input image identity")
        identities.append(identity)
    if [item["subject"] for item in identities] != sorted(audit["subject_ids"]):
        raise RuntimeError("Input identity roster differs from evaluated subjects")
    identity_hash = canonical_hash(identities)
    if audit.get("input_identity_sha256", identity_hash) != identity_hash:
        raise RuntimeError("Input identity digest differs from evaluation audit")
    inference = audit["inference"]
    inference = {"patch_size_dhw": inference["patch_size_dhw"], "overlap_dhw": inference["overlap_dhw"],
                 "gaussian_sigma_scale": inference.get("gaussian_sigma_scale", .125),
                 "amp": inference.get("amp", True), "data_range": inference.get("data_range", 1.0)}
    policy = {key: audit[key] for key in ("dataset", "split", "modality", "scale", "original_manifest_sha256", "metric_protocol", "degradation", "normalization")}
    policy.update(subject_ids=sorted(audit["subject_ids"]), inference=inference, input_identity_sha256=identity_hash)
    digest = canonical_hash(policy)
    if audit.get("comparison_policy_sha256", digest) != digest:
        raise RuntimeError("Comparison policy digest differs from verified evaluation settings")
    return digest


def verify_source_code(item: dict, audit: dict) -> None:
    files = audit.get("source_code_files", {})
    if not files or canonical_hash(files) != audit.get("source_code_sha256"):
        raise RuntimeError("Missing/inconsistent source code provenance")
    if item.get("source_code_sha256") != audit["source_code_sha256"]:
        raise RuntimeError("Source code digest differs from registry")
    if item.get("source_snapshot_manifest"):
        path = Path(item["source_snapshot_manifest"])
        require_hash(path, item.get("source_snapshot_manifest_sha256", ""), "source snapshot manifest")
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if snapshot.get("evaluation_audit_sha256") != item["audit_sha256"] or snapshot.get("source_code_sha256") != audit["source_code_sha256"]:
            raise RuntimeError("Source snapshot is bound to a different audit or source digest")
        archived = snapshot.get("files", {})
        if set(archived) != set(files):
            raise RuntimeError("Source snapshot file roster differs from original audit")
        for original, digest in files.items():
            if archived[original].get("sha256") != digest:
                raise RuntimeError("Source snapshot file identity differs from original audit")
            require_hash(Path(archived[original]["archived_path"]), digest, "archived source code")
    else:
        for path, digest in files.items():
            require_hash(Path(path), digest, "source code")


def training_budget_policy(cfg: dict, audit: dict, submission_mode: bool = False) -> str:
    """No execution-budget inference from a nominal epoch configuration."""
    path = audit.get("training_budget_evidence_path")
    if not path:
        if submission_mode:
            raise RuntimeError("Formal evidence requires independently bound actual training budget/counters")
        return "unknown_diagnostic"
    path = Path(path)
    require_hash(path, audit.get("training_budget_evidence_sha256", ""), "training budget evidence")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if submission_mode and evidence.get("evidence_class") == "diagnostic":
        raise RuntimeError("Formal evidence rejects diagnostic training budgets")
    if submission_mode and (any(token in str(path).upper() for token in FORBIDDEN) or evidence.get("fixture", False) or evidence.get("simulated", False)):
        raise RuntimeError("Formal training budget rejects fixture/simulated evidence")
    if evidence.get("checkpoint_sha256") != audit["checkpoint_sha256"] or evidence.get("source_config_sha256") != cfg["_config_sha256"]:
        raise RuntimeError("Training budget evidence is bound to another model/configuration")
    required = ("epochs_completed", "optimizer_updates", "micro_batches", "effective_global_batch_size",
                "world_size", "batch_size_per_gpu", "gradient_accumulation", "training_manifest_sha256",
                "optimizer", "scheduler", "learning_rate", "weight_decay", "scheduler_steps", "source_log_files")
    if any(key not in evidence for key in required):
        raise RuntimeError("Actual training budget evidence lacks executed counters/settings")
    training = cfg["training"]
    for key in required[:7]:
        if int(evidence[key]) <= 0:
            raise RuntimeError(f"Nonpositive actual training counter: {key}")
    for evidence_key, cfg_key in (("epochs_completed", "epochs"), ("batch_size_per_gpu", "batch_size_per_gpu"),
                                  ("gradient_accumulation", "gradient_accumulation"), ("learning_rate", "learning_rate"), ("weight_decay", "weight_decay")):
        if evidence[evidence_key] != training.get(cfg_key, 1 if cfg_key == "gradient_accumulation" else None):
            raise RuntimeError(f"Executed training {evidence_key} differs from frozen configuration")
    batch = int(evidence["world_size"]) * int(evidence["batch_size_per_gpu"]) * int(evidence["gradient_accumulation"])
    if batch != int(evidence["effective_global_batch_size"]) or batch != int(training.get("target_global_batch_size", batch)):
        raise RuntimeError("Actual global training batch disagrees with frozen target")
    if evidence["optimizer"] != "AdamW" or evidence["scheduler"] != {"name": "CosineAnnealingLR", "T_max": int(training["epochs"])}:
        raise RuntimeError("Actual optimizer/scheduler differ from the verified trainer")
    if int(evidence["scheduler_steps"]) != int(evidence["epochs_completed"]) or int(evidence["optimizer_updates"]) > int(evidence["micro_batches"]):
        raise RuntimeError("Actual optimizer/scheduler counters are inconsistent")
    if len(str(evidence["training_manifest_sha256"])) != 64 or not evidence["source_log_files"]:
        raise RuntimeError("Actual training budget lacks training roster/log provenance")
    for log, digest in evidence["source_log_files"].items():
        if submission_mode and any(token in str(log).upper() for token in FORBIDDEN):
            raise RuntimeError("Formal training budget rejects fixture/simulated source logs")
        log_path = Path(log)
        if not log_path.is_absolute():
            log_path = path.parent / log_path
        require_hash(log_path.resolve(), digest, "training budget source log")
    if training.get("record_actual_budget", False):
        if sha256_file(cfg["data"]["train_manifest"]) != evidence["training_manifest_sha256"]:
            raise RuntimeError("Executed budget training manifest changed")
        logs = [key for key in evidence["source_log_files"] if str(key).endswith('.execution.jsonl')]
        if len(logs) != 1:
            raise RuntimeError("Executed budget requires one independently bound event ledger")
        log = Path(logs[0])
        log = log if log.is_absolute() else path.parent / log
        events = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]
        batches = [r for r in events if r['kind'] == 'micro_batch']
        epochs = [r for r in events if r['kind'] == 'epoch_completed']
        if any(r['kind'] not in {'micro_batch', 'epoch_completed'} for r in events):
            raise RuntimeError("Unknown executed-budget event")
        if [r['epoch'] for r in epochs] != list(range(1, int(evidence['epochs_completed']) + 1)):
            raise RuntimeError("Executed-budget epochs are not complete and ordered")
        counters = dict(micro_batches=len(batches), optimizer_updates=sum(r['optimizer_updated'] for r in batches),
                        optimizer_attempts=sum(r['optimizer_attempted'] for r in batches),
                        skipped_optimizer_updates=sum(r['overflow'] for r in batches),
                        sampled_patches=sum(r['batch_size'] for r in batches), scheduler_steps=len(epochs))
        if any(evidence.get(k) != v for k, v in counters.items()):
            raise RuntimeError("Executed budget counters disagree with independently hashed events")
        if counters['optimizer_attempts'] != counters['optimizer_updates'] + counters['skipped_optimizer_updates']:
            raise RuntimeError("Executed budget optimizer boundaries disagree")
        runtime = json.loads((path.parent / 'runtime.json').read_text(encoding='utf-8'))
        source = json.loads((path.parent / 'training_source/manifest.json').read_text(encoding='utf-8'))
        if canonical_hash(runtime) != evidence.get('runtime_sha256') or canonical_hash(source['files']) != evidence.get('source_code_sha256'):
            raise RuntimeError("Executed training runtime/source digest mismatch")
        expected_sources = {f'training_source/{rel}': digest for rel, digest in source['files'].items()}
        if any(evidence['source_log_files'].get(k) != v for k, v in expected_sources.items()):
            raise RuntimeError("Executed budget lacks archived training source identities")
    # Paths, seed and model/loss choices are intentionally excluded. They are not training budgets.
    declared = {key: value for key, value in training.items() if key not in {"checkpoint_epochs"}}
    data = {key: cfg["data"].get(key) for key in ("train_samples_per_epoch", "patch_size_dhw", "foreground_probability", "augmentation")}
    actual = {key: evidence[key] for key in required if key != "source_log_files"}
    return canonical_hash({"declared": declared, "data": data, "actual": actual})


def validate_method_binding(item: dict, cfg: dict, audit: dict, protocol: dict, submission_mode: bool) -> None:
    group = str(item.get("group", "main")).lower()
    for source in (audit, protocol):
        if "evaluation_group" in source and str(source["evaluation_group"]).lower() != group:
            raise RuntimeError("Evaluation group differs from registered result group")
    if group in {"main", "baselines"}:
        if str(item["method"]).lower() != str(cfg["model"]["name"]).lower():
            raise RuntimeError("Registered method label differs from actual source-config model")
    elif submission_mode or "config_variant_sha256" in audit:
        variant_hash = canonical_hash({"model": cfg["model"], "loss": cfg["loss"]})
        if audit.get("config_variant_sha256") != variant_hash or audit.get("factory_model") != cfg["model"]["name"]:
            raise RuntimeError("Formal ablation label lacks verified model/loss variant binding")


def verify_result(item: dict, submission_mode: bool) -> tuple[pd.DataFrame, dict | None]:
    """Revalidate immutable inputs at collection and table generation."""
    path = Path(item["metrics_csv"])
    require_hash(path, item.get("metrics_sha256", ""), "metrics")
    frame = pd.read_csv(path, dtype={"subject": str})
    if "subject" not in frame or frame.empty or frame.subject.isna().any() or frame.subject.duplicated().any():
        raise RuntimeError(f"Expected one nonempty row per unique subject: {path}")
    if frame.subject.str.strip().eq("").any():
        raise RuntimeError(f"Empty subject ID: {path}")
    audit = None
    if item.get("audit_json"):
        audit_path = Path(item["audit_json"])
        require_hash(audit_path, item.get("audit_sha256", ""), "audit")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if submission_mode:
        sources = [str(item.get(key, "")) for key in ("metrics_csv", "audit_json", "config", "checkpoint", "split_manifest", "protocol", "source_snapshot_manifest")]
        if any(token in source.upper() for source in sources for token in FORBIDDEN):
            raise RuntimeError("Submission mode rejected fixture/simulated/placeholder source")
        if item.get("evidence_class") != "formal" or not audit or audit.get("evidence_class") != "formal":
            raise RuntimeError("Submission mode requires formal evidence with an explicit evaluation audit")
        if audit.get("fixture", False) or audit.get("simulated", False):
            raise RuntimeError("Submission mode rejected fixture/simulated audit")
        if item.get("split") not in {"test", "external"}:
            raise RuntimeError("Submission mode rejects validation/diagnostic evaluations")
    if audit:
        if audit.get("status") != "completed" or audit.get("full_volume") is not True:
            raise RuntimeError("Evaluation audit must certify completed full-volume evaluation")
        if audit.get("metric_protocol", {}).get("unit") not in {"subject_full_volume", "subject-level full3Dvolume"}:
            raise RuntimeError("Evaluation audit must certify subject-level full-volume metrics")
        if int(audit.get("cases", -1)) != len(frame) or set(audit.get("subject_ids", [])) != set(frame.subject):
            raise RuntimeError("Audit cases/subject roster differ from metrics CSV")
        if len(audit.get("subject_ids", [])) != len(frame):
            raise RuntimeError("Duplicate audit subject IDs")
        if audit.get("metrics_sha256") != item["metrics_sha256"]:
            raise RuntimeError("Audit metrics hash differs from registry")
        for key in META:
            expected = normalized_metadata(item[key], key)
            if normalized_metadata(audit.get(key), key) != expected:
                raise RuntimeError(f"Audit {key} differs from run metadata")
            if key not in frame or frame[key].isna().any() or any(normalized_metadata(value, key) != expected for value in frame[key]):
                raise RuntimeError(f"CSV {key} differs from run metadata")
        bindings = (("config", "config_file_sha256", "config_file_sha256"),
                    ("checkpoint", "checkpoint_sha256", "checkpoint_sha256"),
                    ("split_manifest", "original_manifest_sha256", "split_sha256"),
                    ("protocol", "protocol_sha256", "protocol_sha256"))
        for path_key, audit_key, registry_key in bindings:
            if not item.get(path_key) or not item.get(registry_key):
                raise RuntimeError(f"Missing provenance binding: {path_key}/{registry_key}")
            require_hash(Path(item[path_key]), item[registry_key], path_key)
            if audit.get(audit_key) != item[registry_key]:
                raise RuntimeError(f"Audit {audit_key} differs from registry")
        cfg = load_config(item["config"])
        if cfg["_config_sha256"] != item.get("config_sha256") or audit.get("source_config_sha256") != cfg["_config_sha256"]:
            raise RuntimeError("Resolved source configuration SHA256 mismatch")
        for value, key in ((cfg["data"]["modality"], "modality"), (cfg["degradation"]["scale"], "scale"), (cfg["experiment"]["seed"], "seed")):
            if normalized_metadata(value, key) != normalized_metadata(item[key], key):
                raise RuntimeError(f"Configuration {key} differs from run metadata")
        full_roster = set(roster(Path(item["split_manifest"])))
        actual_roster = set(frame.subject)
        protocol = json.loads(Path(item["protocol"]).read_text(encoding="utf-8"))
        validate_method_binding(item, cfg, audit, protocol, submission_mode)
        if submission_mode and full_roster != actual_roster:
            raise RuntimeError("Metrics subject set differs from frozen split manifest")
        if not submission_mode and full_roster != actual_roster:
            selected = protocol.get("subject_ids")
            if not selected or len(selected) != len(set(selected)) or set(selected) != actual_roster or not actual_roster <= full_roster:
                raise RuntimeError("Diagnostic subset must exactly match protocol selection within frozen split roster")
        if submission_mode and protocol.get("subject_ids") is not None:
            raise RuntimeError("Formal evaluation must include the entire frozen original manifest")
        verify_source_code(item, audit)
        for key in ("path_map", "file_inventory", "overlap_audit"):
            if audit.get(f"{key}_path"):
                require_hash(Path(audit[f"{key}_path"]), audit.get(f"{key}_sha256", ""), key)
        policy = comparison_policy(audit, submission_mode)
        if submission_mode and not item.get("comparison_policy_sha256"):
            raise RuntimeError("Formal registry lacks derived comparison policy digest")
        if item.get("comparison_policy_sha256", policy) != policy:
            raise RuntimeError("Comparison policy digest differs from registry")
        budget = training_budget_policy(cfg, audit, submission_mode)
        if item.get("training_budget_sha256", budget) != budget:
            raise RuntimeError("Training budget digest differs from registry")
    for prefix in (item.get("model_prefix", "sr"), item.get("baseline_prefix", "baseline")):
        for region in ("brain", "tumor"):
            for metric in METRICS:
                column = f"{prefix}_{region}_{metric}"
                if column not in frame:
                    raise RuntimeError(f"Missing required metric column {column}: {path}")
                frame[column] = numeric_metric(frame[column], column, allow_missing=region == "tumor")
    availability = item.get("segmentation_available", True)
    if "segmentation_available" in frame:
        flags = frame.segmentation_available.map(as_bool)
        aggregate = bool(flags.all()) if flags.all() or not flags.any() else "mixed"
        if availability != aggregate or (audit and audit.get("segmentation_available") != aggregate):
            raise RuntimeError("Segmentation availability metadata disagree")
    else:
        if availability == "mixed":
            raise RuntimeError("Mixed segmentation availability requires per-subject flags")
        flags = pd.Series(as_bool(availability), index=frame.index)
        if audit and audit.get("segmentation_available") != availability:
            raise RuntimeError("Segmentation availability metadata disagree")
    tumor_columns = [f"{prefix}_tumor_{metric}" for prefix in (item.get("model_prefix", "sr"), item.get("baseline_prefix", "baseline")) for metric in METRICS]
    if frame.loc[~flags, tumor_columns].notna().any().any():
        raise RuntimeError("Tumor metrics must be N/A when segmentation is unavailable")
    return frame, audit


def collect(runs: pd.DataFrame, base: Path, allow_incomplete: bool = False) -> dict:
    completed, missing, ids = [], [], set()
    for row in runs.to_dict("records"):
        run_id = str(row["run_id"])
        split = str(row.get("split")) if present(row.get("split")) else "test"
        result_id = str(row.get("evaluation_id")) if present(row.get("evaluation_id")) else f"{run_id}:{split}"
        if result_id in ids:
            raise RuntimeError(f"Duplicate evaluation id: {result_id}; use explicit evaluation_id")
        ids.add(result_id)
        if present(row.get("metrics_csv")):
            metrics = resolve_path(row["metrics_csv"], base)
        else:
            output = resolve_path(row["output_dir"], base)
            candidates = sorted(output.glob(f"{split}_*_subject_metrics.csv"))
            if len(candidates) > 1:
                raise RuntimeError(f"Ambiguous legacy evaluation files for {run_id}; supply metrics_csv explicitly")
            metrics = candidates[0] if candidates else None
        if metrics is None or not metrics.is_file():
            missing.append(result_id)
            continue
        config = resolve_path(row["config"], base)
        cfg = load_config(config)
        audit_path = resolve_path(row["audit_json"], base) if present(row.get("audit_json")) else None
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path else {}
        dataset = row.get("evaluation_dataset") if present(row.get("evaluation_dataset")) else row["dataset"]
        item = {"id": result_id, "run_id": run_id, "group": row.get("group", "main"),
                "dataset": str(dataset), "training_dataset": str(cfg["data"].get("dataset", "unknown")),
                "split": split, "modality": str(row["modality"]), "scale": int(row["scale"]),
                "method": str(row["method"]), "seed": int(row["seed"]),
                "training_regime": str(row["training_regime"]) if present(row.get("training_regime")) else "shared",
                "metrics_csv": str(metrics.resolve()), "metrics_sha256": sha256_file(metrics),
                "config": str(config), "config_file_sha256": sha256_file(config), "config_sha256": cfg["_config_sha256"],
                "evidence_class": audit.get("evidence_class", "diagnostic"),
                "segmentation_available": audit.get("segmentation_available", True),
                "model_prefix": "sr", "baseline_prefix": "baseline"}
        if audit_path:
            item.update(audit_json=str(audit_path), audit_sha256=sha256_file(audit_path), source_code_sha256=audit.get("source_code_sha256", ""))
            if present(row.get("source_snapshot_manifest")):
                if not present(row.get("source_snapshot_manifest_sha256")):
                    raise RuntimeError("Source snapshot requires an explicit run-manifest SHA256 binding")
                item.update(source_snapshot_manifest=str(resolve_path(row["source_snapshot_manifest"], base)),
                            source_snapshot_manifest_sha256=str(row["source_snapshot_manifest_sha256"]))
            for path_key, audit_path_key, hash_key, audit_hash_key in (
                ("checkpoint", "checkpoint_path", "checkpoint_sha256", "checkpoint_sha256"),
                ("split_manifest", "original_manifest_path", "split_sha256", "original_manifest_sha256"),
                ("protocol", "protocol_path", "protocol_sha256", "protocol_sha256")):
                value = row.get(path_key) if present(row.get(path_key)) else audit.get(audit_path_key)
                if value:
                    item[path_key] = str(resolve_path(value, base))
                item[hash_key] = audit.get(audit_hash_key, "")
        item["comparison_policy_sha256"] = comparison_policy(audit, item["evidence_class"] == "formal") if audit else "unbound_diagnostic"
        verify_result(item, submission_mode=item["evidence_class"] == "formal")
        item["training_budget_sha256"] = training_budget_policy(cfg, audit, item["evidence_class"] == "formal") if audit else "unknown_diagnostic"
        completed.append(item)
    if missing and not allow_incomplete:
        raise RuntimeError(f"Missing evaluation metrics for {len(missing)} runs; first ten: {missing[:10]}")
    return {"schema_version": 3, "completed_results": completed, "missing_runs": missing,
            "submission_eligible": bool(completed) and not missing and all(item["evidence_class"] == "formal" for item in completed),
            "usage": "Formal eligibility is rechecked by build_tables --submission-mode; diagnostics are NOT_FOR_SUBMISSION."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--out", default="COMPLETED_RESULTS.yaml")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    source = Path(args.run_manifest).resolve()
    payload = collect(pd.read_csv(source), source.parent, args.allow_incomplete)
    payload.update(source_run_manifest=str(source), source_run_manifest_sha256=sha256_file(source))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"Collected {len(payload['completed_results'])} evaluations; missing={len(payload['missing_runs'])}; submission_eligible={payload['submission_eligible']}; out={out}")


if __name__ == "__main__":
    main()
