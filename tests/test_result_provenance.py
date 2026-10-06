"""Synthetic technical checks; fixtures must never be accepted as paper evidence."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from collect_results import collect, verify_result, training_budget_policy, validate_method_binding
from build_tables import bootstrap_ci, compact_table, load_long, summarize, validate_seed_and_subject_sets
from fdmrnet.config import canonical_hash, load_config, sha256_file


def make_evaluation(root: Path, seed: int = 2025, subjects=("case_a", "case_b")) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump({
        "experiment": {"id": "technical_fixture", "seed": seed},
        "data": {"dataset": "BraTS2021", "modality": "t1", "patch_size_dhw": [8, 8, 8]},
        "degradation": {"axis": "depth", "scale": 4}, "model": {"name": "fdmrnet"},
        "loss": {}, "training": {}, "evaluation": {},
    }), encoding="utf-8")
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic-not-a-model")
    manifest = root / "subjects.csv"
    pd.DataFrame({"subject": subjects}).to_csv(manifest, index=False)
    protocol = root / "protocol.json"
    protocol.write_text(json.dumps({"fixture": True, "split": "external", "seed": seed}), encoding="utf-8")
    code = root / "source.py"
    code.write_text("# synthetic technical fixture\n", encoding="utf-8")
    metrics = root / "external_final_subject_metrics.csv"
    rows = []
    for index, subject in enumerate(subjects):
        row = {"subject": subject, "dataset": "BraTS2023", "split": "external", "modality": "t1",
               "scale": 4, "method": "fdmrnet", "seed": seed, "segmentation_available": False}
        for prefix in ("sr", "baseline"):
            for metric, value in {"psnr": 30 + index, "ssim": .8, "nmse": .02, "hfen": .1}.items():
                row[f"{prefix}_brain_{metric}"] = value
                row[f"{prefix}_tumor_{metric}"] = "N/A"
        rows.append(row)
    pd.DataFrame(rows).to_csv(metrics, index=False)
    code_files = {str(code): sha256_file(code)}
    audit = root / "audit.json"
    audit.write_text(json.dumps({
        "status": "completed", "full_volume": True,
        "metric_protocol": {"unit": "subject-level full3Dvolume"},
        "subject_ids": list(subjects), "cases": len(subjects), "dataset": "BraTS2023",
        "split": "external", "modality": "t1", "scale": 4, "method": "fdmrnet", "seed": seed,
        "segmentation_available": False, "evidence_class": "diagnostic", "fixture": True,
        "metrics_sha256": sha256_file(metrics), "source_config_sha256": load_config(config)["_config_sha256"],
        "config_file_sha256": sha256_file(config), "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint), "original_manifest_path": str(manifest),
        "original_manifest_sha256": sha256_file(manifest), "protocol_path": str(protocol),
        "protocol_sha256": sha256_file(protocol), "source_code_files": code_files,
        "source_code_sha256": canonical_hash(code_files),
        "degradation": {"axis": "depth", "scale": 4, "sigma_vox": 1.6985},
        "normalization": {"scope": "HR nonzero foreground", "clip_z": 5},
        "inference": {"patch_size_dhw": [8, 8, 8], "overlap_dhw": [2, 2, 2], "amp": False},
        "file_identity_audit": [{"subject": subject, "image": {
            "original_path": f"/original/{subject}.nii.gz", "sha256": "a" * 64,
            "shape_xyz": [8, 8, 8], "affine": np.eye(4).tolist(),
        }} for subject in subjects],
    }), encoding="utf-8")
    return {"run_id": f"technical_seed{seed}", "group": "main", "dataset": "BraTS2021",
            "evaluation_dataset": "BraTS2023", "split": "external", "modality": "t1", "scale": 4,
            "method": "fdmrnet", "seed": seed, "config": str(config), "metrics_csv": str(metrics),
            "audit_json": str(audit), "output_dir": str(root)}


def update_audit(row, **changes):
    path = Path(row["audit_json"])
    audit = json.loads(path.read_text())
    audit.update(changes)
    path.write_text(json.dumps(audit), encoding="utf-8")


def update_metrics(row, frame):
    path = Path(row["metrics_csv"])
    frame.to_csv(path, index=False)
    update_audit(row, metrics_sha256=sha256_file(path))


def test_external_dataset_and_na_survive_four_metric_table(tmp_path):
    runs = [make_evaluation(tmp_path / str(seed), seed) for seed in (2025, 2026, 2027)]
    registry = collect(pd.DataFrame(runs), tmp_path)
    assert registry["submission_eligible"] is False
    assert registry["completed_results"][0]["dataset"] == "BraTS2023"
    assert registry["completed_results"][0]["training_dataset"] == "BraTS2021"
    long = load_long(registry, False)
    validate_seed_and_subject_sets(long, 3)
    summary = summarize(long)
    tumor = summary[summary.region == "tumor"]
    assert (tumor.valid_subjects == 0).all() and tumor.subject_mean.isna().all()
    table = compact_table(summary, "BraTS2023", {"main"})
    assert {f"x4_{metric}" for metric in ("PSNR", "SSIM", "NMSE", "HFEN")} <= set(table)
    assert set(long.split) == {"external"}


def test_fixture_cannot_be_reclassified_formal(tmp_path):
    row = make_evaluation(tmp_path)
    update_audit(row, evidence_class="formal")
    with pytest.raises(RuntimeError, match="fixture/simulated"):
        collect(pd.DataFrame([row]), tmp_path)


def test_diagnostic_cannot_enter_submission(tmp_path):
    registry = collect(pd.DataFrame([make_evaluation(tmp_path)]), tmp_path)
    item = registry["completed_results"][0]
    with pytest.raises(RuntimeError, match="formal evidence|fixture/simulated"):
        verify_result(item, True)


@pytest.mark.parametrize("change", ["hash", "checkpoint", "roster", "metadata", "incomplete", "unit"])
def test_provenance_failures_are_stop_conditions(tmp_path, change):
    row = make_evaluation(tmp_path)
    if change == "hash":
        registry = collect(pd.DataFrame([row]), tmp_path)
        Path(row["metrics_csv"]).write_text("changed", encoding="utf-8")
        with pytest.raises(RuntimeError, match="SHA256 mismatch"):
            load_long(registry, False)
        return
    if change == "checkpoint":
        (tmp_path / "checkpoint.pt").write_bytes(b"changed")
    elif change == "roster":
        update_audit(row, subject_ids=["case_a", "wrong"])
    elif change == "metadata":
        update_audit(row, scale=2)
    elif change == "incomplete":
        update_audit(row, status="running")
    elif change == "unit":
        update_audit(row, metric_protocol={"unit": "center_patch"})
    with pytest.raises(RuntimeError):
        collect(pd.DataFrame([row]), tmp_path)


@pytest.mark.parametrize("change", ["missing_column", "nan_brain", "bad_string", "duplicate", "invented_tumor"])
def test_invalid_metrics_rejected(tmp_path, change):
    row = make_evaluation(tmp_path)
    frame = pd.read_csv(row["metrics_csv"])
    if change == "missing_column":
        frame = frame.drop(columns="sr_brain_hfen")
    elif change == "nan_brain":
        frame.loc[0, "sr_brain_psnr"] = np.nan
    elif change == "bad_string":
        frame["sr_brain_nmse"] = "typo"
    elif change == "duplicate":
        frame.loc[1, "subject"] = frame.loc[0, "subject"]
    else:
        frame["sr_tumor_psnr"] = 0.0
    update_metrics(row, frame)
    with pytest.raises(RuntimeError):
        collect(pd.DataFrame([row]), tmp_path)


def test_legacy_latest_mtime_is_never_chosen(tmp_path):
    row = make_evaluation(tmp_path)
    frame = pd.read_csv(row["metrics_csv"])
    frame.to_csv(tmp_path / "external_another_subject_metrics.csv", index=False)
    row.pop("metrics_csv")
    with pytest.raises(RuntimeError, match="Ambiguous legacy"):
        collect(pd.DataFrame([row]), tmp_path)


def test_same_subject_union_cannot_hide_seed_roster_mismatch(tmp_path):
    runs = [make_evaluation(tmp_path / str(seed), seed) for seed in (2025, 2026, 2027)]
    long = load_long(collect(pd.DataFrame(runs), tmp_path), False)
    # Union remains case_a/case_b, but one seed is missing case_b.
    long = long[~((long.seed == 2026) & (long.subject == "case_b"))]
    with pytest.raises(RuntimeError, match="Subject set mismatch between seeds"):
        validate_seed_and_subject_sets(long, 3)


def test_exact_seed_count_and_method_seed_matching(tmp_path):
    runs = [make_evaluation(tmp_path / str(seed), seed) for seed in (2025, 2026, 2027)]
    long = load_long(collect(pd.DataFrame(runs), tmp_path), False)
    with pytest.raises(RuntimeError, match="exactly 3 seeds"):
        validate_seed_and_subject_sets(long[long.seed != 2027], 3)
    long.loc[long.method == "trilinear", "seed"] += 1
    with pytest.raises(RuntimeError, match="Seed IDs mismatch"):
        validate_seed_and_subject_sets(long, 3)


def test_cli_diagnostic_output_is_visibly_marked(tmp_path):
    runs = [make_evaluation(tmp_path / str(seed), seed) for seed in (2025, 2026, 2027)]
    registry = collect(pd.DataFrame(runs), tmp_path)
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    out = tmp_path / "tables"
    command = [sys.executable, str(SCRIPTS / "build_tables.py"), "--registry", str(registry_path), "--out", str(out)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (out / "NOT_FOR_SUBMISSION.txt").is_file()
    assert "NOT_FOR_SUBMISSION" in (out / "Table_II_BraTS2023.md").read_text()
    rejected = subprocess.run(command + ["--submission-mode"], capture_output=True, text=True)
    assert rejected.returncode != 0


def test_unique_per_run_protocols_share_comparison_settings(tmp_path):
    runs = [make_evaluation(tmp_path / str(seed), seed) for seed in (2025, 2026, 2027)]
    registry = collect(pd.DataFrame(runs), tmp_path)
    items = registry["completed_results"]
    assert len({item["protocol_sha256"] for item in items}) == 3
    assert len({item["comparison_policy_sha256"] for item in items}) == 1
    validate_seed_and_subject_sets(load_long(registry, False), 3)


def test_changed_inference_settings_cannot_be_merged(tmp_path):
    runs = [make_evaluation(tmp_path / str(seed), seed) for seed in (2025, 2026, 2027)]
    update_audit(runs[-1], inference={"patch_size_dhw": [16, 8, 8], "overlap_dhw": [2, 2, 2], "amp": False})
    registry = collect(pd.DataFrame(runs), tmp_path)
    with pytest.raises(RuntimeError, match="comparison_policy_sha256 mismatch"):
        validate_seed_and_subject_sets(load_long(registry, False), 3)


def test_diagnostic_subset_is_bound_and_never_formal(tmp_path):
    row = make_evaluation(tmp_path, subjects=("case_a",))
    manifest = tmp_path / "subjects.csv"
    pd.DataFrame({"subject": ["case_a", "case_b"]}).to_csv(manifest, index=False)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"fixture": True, "split": "external", "subject_ids": ["case_a"]}), encoding="utf-8")
    update_audit(row, original_manifest_sha256=sha256_file(manifest), protocol_sha256=sha256_file(protocol))
    registry = collect(pd.DataFrame([row]), tmp_path)
    assert not registry["submission_eligible"]
    long = load_long(registry, False)
    assert set(long.subject) == {"case_a"}
    assert not compact_table(summarize(long), "BraTS2023", {"main"}).empty
    with pytest.raises(RuntimeError, match="formal evidence|fixture/simulated"):
        verify_result(registry["completed_results"][0], True)
    protocol.write_text(json.dumps({"fixture": True, "split": "external", "subject_ids": ["case_b"]}), encoding="utf-8")
    update_audit(row, protocol_sha256=sha256_file(protocol))
    with pytest.raises(RuntimeError, match="Diagnostic subset"):
        collect(pd.DataFrame([row]), tmp_path)


def test_exact_source_snapshot_survives_live_code_change(tmp_path):
    row = make_evaluation(tmp_path)
    audit = json.loads(Path(row["audit_json"]).read_text())
    original = tmp_path / "source.py"
    archived = tmp_path / "archived_source.py"
    archived.write_bytes(original.read_bytes())
    snapshot = tmp_path / "source_snapshot_manifest.json"
    snapshot.write_text(json.dumps({"evaluation_audit_sha256": sha256_file(row["audit_json"]),
        "source_code_sha256": audit["source_code_sha256"],
        "files": {str(original): {"archived_path": str(archived), "sha256": sha256_file(archived)}},
    }), encoding="utf-8")
    original.write_text("# later working-tree edit\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source code SHA256"):
        collect(pd.DataFrame([row]), tmp_path)
    row.update(source_snapshot_manifest=str(snapshot), source_snapshot_manifest_sha256=sha256_file(snapshot))
    registry = collect(pd.DataFrame([row]), tmp_path)
    assert not load_long(registry, False).empty
    archived.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="archived source code SHA256"):
        load_long(registry, False)


def test_snapshot_cannot_be_bound_to_another_audit(tmp_path):
    row = make_evaluation(tmp_path)
    audit = json.loads(Path(row["audit_json"]).read_text())
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"evaluation_audit_sha256": "0" * 64,
        "source_code_sha256": audit["source_code_sha256"], "files": {}}), encoding="utf-8")
    row.update(source_snapshot_manifest=str(snapshot), source_snapshot_manifest_sha256=sha256_file(snapshot))
    with pytest.raises(RuntimeError, match="different audit"):
        collect(pd.DataFrame([row]), tmp_path)


def test_arbitrary_method_label_cannot_rename_checkpoint(tmp_path):
    row = make_evaluation(tmp_path)
    frame = pd.read_csv(row["metrics_csv"])
    frame["method"] = "rdn3d"
    update_metrics(row, frame)
    update_audit(row, method="rdn3d")
    row["method"] = "rdn3d"
    with pytest.raises(RuntimeError, match="method label differs"):
        collect(pd.DataFrame([row]), tmp_path)


def add_budget(row, updates=1000):
    root = Path(row["config"]).parent
    config = yaml.safe_load(Path(row["config"]).read_text())
    config["training"] = {"epochs": 2, "batch_size_per_gpu": 1, "target_global_batch_size": 1,
                          "gradient_accumulation": 1, "learning_rate": .0001, "weight_decay": .001}
    Path(row["config"]).write_text(yaml.safe_dump(config), encoding="utf-8")
    cfg = load_config(row["config"])
    log = root / "training_fixture.json"
    log.write_text('{"fixture":true}', encoding="utf-8")
    budget = root / "budget_fixture.json"
    budget.write_text(json.dumps({"checkpoint_sha256": sha256_file(root / "checkpoint.pt"),
        "source_config_sha256": cfg["_config_sha256"], "epochs_completed": 2,
        "optimizer_updates": updates, "micro_batches": 2000, "effective_global_batch_size": 1,
        "world_size": 1, "batch_size_per_gpu": 1, "gradient_accumulation": 1,
        "training_manifest_sha256": "b" * 64, "optimizer": "AdamW",
        "scheduler": {"name": "CosineAnnealingLR", "T_max": 2},
        "learning_rate": .0001, "weight_decay": .001, "scheduler_steps": 2,
        "source_log_files": {str(log): sha256_file(log)},
    }), encoding="utf-8")
    update_audit(row, config_file_sha256=sha256_file(row["config"]), source_config_sha256=cfg["_config_sha256"],
                 training_budget_evidence_path=str(budget), training_budget_evidence_sha256=sha256_file(budget))


def test_different_actual_budget_is_not_shared_budget(tmp_path):
    runs = [make_evaluation(tmp_path / str(seed), seed) for seed in (2025, 2026, 2027)]
    for row in runs:
        add_budget(row, updates=1000 if row["seed"] != 2027 else 1001)
    registry = collect(pd.DataFrame(runs), tmp_path)
    with pytest.raises(RuntimeError, match="training_budget_sha256 mismatch"):
        validate_seed_and_subject_sets(load_long(registry, False), 3)


def test_unknown_actual_budget_is_a_formal_blocker(tmp_path):
    row = make_evaluation(tmp_path)
    cfg = load_config(row["config"])
    audit = json.loads(Path(row["audit_json"]).read_text())
    assert training_budget_policy(cfg, audit, False) == "unknown_diagnostic"
    with pytest.raises(RuntimeError, match="actual training budget"):
        training_budget_policy(cfg, audit, True)


def test_one_subject_has_no_inferential_ci():
    assert all(np.isnan(value) for value in bootstrap_ci(np.array([30.])))


def test_relative_budget_log_paths_are_relative_to_evidence_file(tmp_path):
    row = make_evaluation(tmp_path)
    add_budget(row)
    path = tmp_path / "budget_fixture.json"
    evidence = json.loads(path.read_text())
    evidence["source_log_files"] = {"training_fixture.json": sha256_file(tmp_path / "training_fixture.json")}
    path.write_text(json.dumps(evidence), encoding="utf-8")
    update_audit(row, training_budget_evidence_sha256=sha256_file(path))
    registry = collect(pd.DataFrame([row]), tmp_path)
    assert registry["completed_results"][0]["training_budget_sha256"] != "unknown_diagnostic"


def test_config_backed_ablation_factory_binding():
    cfg = {"model": {"name": "fdmrnet", "use_attention": False}, "loss": {"lambda_rec": 1.}}
    item = {"group": "ablations", "method": "no_attention"}
    audit = {"evaluation_group": "ablations", "factory_model": "fdmrnet",
             "config_variant_sha256": canonical_hash({"model": cfg["model"], "loss": cfg["loss"]})}
    # This tests a binding boundary, without promoting a synthetic evaluation to formal.
    validate_method_binding(item, cfg, audit, {"evaluation_group": "ablations"}, True)
    with pytest.raises(RuntimeError, match="group differs"):
        validate_method_binding({"group": "main", "method": "fdmrnet"}, cfg, audit, {}, True)
    with pytest.raises(RuntimeError, match="variant binding"):
        validate_method_binding(item, cfg, audit | {"config_variant_sha256": "0" * 64}, {}, True)


def test_ambiguous_group_rows_cannot_pick_a_table_value(tmp_path):
    registry = collect(pd.DataFrame([make_evaluation(tmp_path)]), tmp_path)
    summary = summarize(load_long(registry, False))
    duplicate = summary[summary.method == "fdmrnet"].copy()
    duplicate["group"] = "baselines"
    with pytest.raises(RuntimeError, match="Ambiguous duplicate"):
        compact_table(pd.concat([summary, duplicate]), "BraTS2023", {"main", "baselines"})
