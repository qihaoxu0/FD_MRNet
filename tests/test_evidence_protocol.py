"""Synthetic 8^3 engineering fixtures only; these never create paper evidence."""
import csv
import json
from datetime import datetime, timezone

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from fdmrnet.config import load_config, sha256_file
from fdmrnet.evaluation.evidence_protocol import (
    ReadOnlyPathResolver, _validate_fresh_overlap, build_file_inventory, evaluate_evidence_protocol,
)
from fdmrnet.models import build_model


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


@pytest.fixture
def bundle(tmp_path):
    actual = tmp_path / "moved"
    actual.mkdir()
    rng = np.random.default_rng(2025)
    image_path = actual / "subject_t1.nii.gz"
    seg_path = actual / "subject_seg.nii.gz"
    nib.save(nib.Nifti1Image(rng.uniform(1, 20, (8, 8, 8)).astype(np.float32), np.eye(4)), image_path)
    seg = np.zeros((8, 8, 8), dtype=np.int16)
    seg[2:6, 2:6, 2:6] = 1
    nib.save(nib.Nifti1Image(seg, np.eye(4)), seg_path)
    original_image, original_seg = "/old/data/subject_t1.nii.gz", "/old/data/subject_seg.nii.gz"
    manifest = tmp_path / "original_val.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["subject", "t1", "t2", "seg"])
        writer.writeheader()
        writer.writerow({"subject": "fixture_subject", "t1": original_image, "t2": "", "seg": original_seg})
    path_map = tmp_path / "path_map.json"
    dump_json(path_map, {"mappings": [{"source_prefix": "/old/data", "target_prefix": str(actual)}]})
    config_path = tmp_path / "resolved_config.yaml"
    config = {
        "experiment": {"id": "engineering_fixture", "seed": 2025, "output_dir": "unused"},
        "data": {"dataset": "BraTS2021", "modality": "t1", "patch_size_dhw": [8, 8, 8],
                 "clip_z": 5.0, "val_manifest": str(manifest)},
        "degradation": {"axis": "depth", "scale": 2, "sigma_vox": 0.8493,
                        "sample_offset": "center", "noise_kind": "none", "noise_std": 0.0},
        "model": {"name": "edsr3d", "channels": 4, "num_blocks": 1},
        "loss": {}, "training": {"epochs": 300},
        "evaluation": {"patch_size_dhw": [8, 8, 8], "overlap_dhw": [2, 2, 2], "amp": False, "data_range": 1.0},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    cfg = load_config(config_path)
    checkpoint = tmp_path / "candidate_500.pt"
    torch.manual_seed(2025)
    torch.save({"model": build_model(config["model"]).state_dict(), "config": config,
                "config_sha256": cfg["_config_sha256"], "manifest_hashes": {"val": sha256_file(manifest)},
                "epoch": 0, "micro_step": 500}, checkpoint)
    inventory_path = tmp_path / "file_inventory.json"
    dump_json(inventory_path, build_file_inventory(manifest, path_map, "t1"))
    protocol = {
        "schema_version": 1, "protocol_id": "synthetic_engineering_v1", "fixture": True,
        "evidence_class": "diagnostic", "split": "val", "dataset": "BraTS2021",
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "source_config": {"path": str(config_path), "sha256": cfg["_config_sha256"]},
        "original_manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
        "path_map": {"path": str(path_map), "sha256": sha256_file(path_map)},
        "file_inventory": {"path": str(inventory_path), "sha256": sha256_file(inventory_path)},
        "subject_ids": None,
        "freeze": {"frozen_at": datetime.now(timezone.utc).isoformat(), "checkpoint_selection_rule": "engineering_candidate_only",
                   "manifest_binding": "bound_at_training", "no_test_tuning": True},
    }
    protocol_path = tmp_path / "protocol.json"
    dump_json(protocol_path, protocol)
    return {"protocol_path": protocol_path, "protocol": protocol, "checkpoint": checkpoint,
            "image": image_path, "manifest": manifest, "config": config_path,
            "inventory": inventory_path, "path_map": path_map, "output": tmp_path / "outputs"}


def rewrite(bundle):
    dump_json(bundle["protocol_path"], bundle["protocol"])


def run(bundle, tag="fixture"):
    return evaluate_evidence_protocol(bundle["protocol_path"], bundle["output"], tag, device="cpu", threads=1)


def test_diagnostic_full_volume_preserves_source_bytes(bundle):
    before = {key: sha256_file(bundle[key]) for key in ("checkpoint", "manifest", "config", "image")}
    output = run(bundle)
    rows = list(csv.DictReader(output.open()))
    audit = json.loads((bundle["output"] / "fixture_audit.json").read_text())
    assert len(rows) == 1 and rows[0]["evidence_class"] == "diagnostic"
    assert audit["status"] == "completed" and audit["full_volume"] is True and audit["fixture"] is True
    assert audit["metric_protocol"]["unit"] == "subject-level full3Dvolume"
    assert audit["cases_audit"][0]["target_shape_dhw"] == [8, 8, 8]
    assert audit["checkpoint_training_metadata"]["micro_step"] == 500
    assert audit["metrics_sha256"] == sha256_file(output)
    assert audit["source_code_files"] and audit["executed_source_code_files"]
    assert all("fixture_source_snapshot" in path and sha256_file(path) == expected
               for path, expected in audit["source_code_files"].items())
    assert audit["training_budget"]["checkpoint_observations"]["micro_step"] == 500
    assert audit["training_budget"]["verification"].startswith("not_verified_diagnostic")
    assert before == {key: sha256_file(bundle[key]) for key in before}
    with pytest.raises(RuntimeError, match="overwrite"):
        run(bundle)


def test_changed_image_bytes_refused(bundle):
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8), np.float32), np.eye(4)), bundle["image"])
    with pytest.raises(RuntimeError, match="Image identity hash mismatch"):
        run(bundle)
    assert not list(bundle["output"].glob("*_subject_metrics.csv"))


def test_wrong_inventory_geometry_refused(bundle):
    inventory = json.loads(bundle["inventory"].read_text())
    inventory["files"]["/old/data/subject_t1.nii.gz"]["shape_xyz"] = [7, 8, 8]
    dump_json(bundle["inventory"], inventory)
    bundle["protocol"]["file_inventory"]["sha256"] = sha256_file(bundle["inventory"])
    rewrite(bundle)
    with pytest.raises(RuntimeError, match="geometry shape mismatch"):
        run(bundle)


def test_changed_source_config_refused(bundle):
    cfg = yaml.safe_load(bundle["config"].read_text())
    cfg["experiment"]["seed"] = 2026
    bundle["config"].write_text(yaml.safe_dump(cfg))
    with pytest.raises(RuntimeError, match="canonical hash mismatch"):
        run(bundle)


def test_checkpoint_hash_and_bound_manifest_hash_refused(bundle):
    bundle["protocol"]["checkpoint"]["sha256"] = "0" * 64
    rewrite(bundle)
    with pytest.raises(RuntimeError, match="checkpoint hash mismatch"):
        run(bundle, "wrong_checkpoint")
    payload = torch.load(bundle["checkpoint"], weights_only=False)
    payload["manifest_hashes"]["val"] = "0" * 64
    torch.save(payload, bundle["checkpoint"])
    bundle["protocol"]["checkpoint"]["sha256"] = sha256_file(bundle["checkpoint"])
    rewrite(bundle)
    with pytest.raises(RuntimeError, match="checkpoint-bound hash"):
        run(bundle, "wrong_manifest")


def test_strict_state_loading_refuses_missing_parameter(bundle):
    payload = torch.load(bundle["checkpoint"], weights_only=False)
    payload["model"].pop(next(iter(payload["model"])))
    torch.save(payload, bundle["checkpoint"])
    bundle["protocol"]["checkpoint"]["sha256"] = sha256_file(bundle["checkpoint"])
    rewrite(bundle)
    with pytest.raises(RuntimeError, match="Missing key"):
        run(bundle)


def test_external_fixture_records_actual_dataset_and_na(bundle):
    records = list(csv.DictReader(bundle["manifest"].open()))
    records[0]["seg"] = ""
    external = bundle["manifest"].with_name("external.csv")
    with external.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    bundle["protocol"]["original_manifest"] = {"path": str(external), "sha256": sha256_file(external)}
    bundle["protocol"].update({"split": "external", "dataset": "BraTS2023", "external_eval_without_seg": True})
    bundle["protocol"]["freeze"]["manifest_binding"] = "not_bound_at_training"
    dump_json(bundle["inventory"], build_file_inventory(external, bundle["path_map"], "t1"))
    bundle["protocol"]["file_inventory"]["sha256"] = sha256_file(bundle["inventory"])
    rewrite(bundle)
    output = run(bundle)
    row = next(csv.DictReader(output.open()))
    audit = json.loads((bundle["output"] / "fixture_audit.json").read_text())
    assert row["dataset"] == "BraTS2023" and row["segmentation_available"] == "False"
    assert row["sr_tumor_psnr"] == "N/A" and row["sr_tumor_valid_n"] == "N/A"
    assert row["tumor_metrics"] == "N/A: segmentation unavailable"
    assert audit["training_dataset"] == "BraTS2021" and audit["manifest_binding"] == "not_bound_at_training"
    assert audit["segmentation_available"] is False


def test_candidate_cannot_be_promoted_to_formal(bundle):
    bundle["protocol"].update({"evidence_class": "formal", "fixture": False, "split": "test"})
    bundle["protocol"]["freeze"].update({"checkpoint_selection_rule": "final_epoch", "manifest_binding": "not_bound_at_training"})
    rewrite(bundle)
    with pytest.raises(RuntimeError, match="completion of the source-config epoch budget"):
        run(bundle)


def test_empty_tumor_is_na_and_keeps_true_zero_voxel_count(bundle):
    seg_path = bundle["image"].with_name("subject_seg.nii.gz")
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.int16), np.eye(4)), seg_path)
    dump_json(bundle["inventory"], build_file_inventory(bundle["manifest"], bundle["path_map"], "t1"))
    bundle["protocol"]["file_inventory"]["sha256"] = sha256_file(bundle["inventory"])
    rewrite(bundle)
    row = next(csv.DictReader(run(bundle).open()))
    assert row["segmentation_available"] == "True"
    assert row["tumor_metrics"] == "N/A: empty tumor ROI"
    assert row["sr_tumor_psnr"] == "N/A" and row["baseline_tumor_hfen"] == "N/A"
    assert row["sr_tumor_valid_n"] == "0"


def test_method_cannot_relabel_checkpoint_factory(bundle):
    bundle["protocol"]["method"] = "rdn3d"
    rewrite(bundle)
    with pytest.raises(RuntimeError, match="model factory"):
        run(bundle)


def test_ablation_requires_source_config_variant_binding(bundle):
    bundle["protocol"].update({"evaluation_group": "ablations", "method": "without_component"})
    rewrite(bundle)
    with pytest.raises(RuntimeError, match="variant binding"):
        run(bundle)


def test_legacy_overlap_report_cannot_be_upgraded_by_counts(bundle):
    legacy = {"full_voxel_hash_performed": True, "subject_id_overlap": [],
              "voxel_identical_overlap": [], "independent_external_cohort": True,
              "dataset_a_subjects": 1251, "dataset_b_subjects": 219}
    with pytest.raises(RuntimeError, match="legacy counts do not qualify"):
        _validate_fresh_overlap(legacy, bundle["protocol"], bundle["protocol_path"].parent,
                                bundle["manifest"], {}, {})


def test_epoch_label_without_verified_actual_budget_is_not_formal(bundle):
    payload = torch.load(bundle["checkpoint"], weights_only=False)
    payload["epoch"] = 300
    payload.pop("micro_step")
    torch.save(payload, bundle["checkpoint"])
    bundle["protocol"]["checkpoint"]["sha256"] = sha256_file(bundle["checkpoint"])
    bundle["protocol"].update({"evidence_class": "formal", "fixture": False, "split": "test"})
    bundle["protocol"]["freeze"].update({"checkpoint_selection_rule": "final_epoch", "manifest_binding": "not_bound_at_training",
                                        "never_used_for_training_tuning_or_selection": True, "late_binding_reason": "technical fixture only"})
    rewrite(bundle)
    with pytest.raises(RuntimeError, match="independently hashed training_budget_evidence"):
        run(bundle)


def test_nominal_epoch_does_not_hide_500_step_candidate(bundle):
    payload = torch.load(bundle["checkpoint"], weights_only=False)
    payload["epoch"] = 300
    torch.save(payload, bundle["checkpoint"])
    bundle["protocol"]["checkpoint"]["sha256"] = sha256_file(bundle["checkpoint"])
    bundle["protocol"].update({"evidence_class": "formal", "fixture": False, "split": "test"})
    bundle["protocol"]["freeze"].update({"checkpoint_selection_rule": "final_epoch", "manifest_binding": "not_bound_at_training"})
    rewrite(bundle)
    with pytest.raises(RuntimeError, match="500-step engineering candidates"):
        run(bundle)


def test_prefix_boundaries_and_traversal(tmp_path):
    resolver = ReadOnlyPathResolver([{"source_prefix": "/old/data", "target_prefix": str(tmp_path)}])
    assert resolver.locate("/old/data/a.nii.gz") == tmp_path / "a.nii.gz"
    assert resolver.locate("/old/database/a.nii.gz") == __import__("pathlib").Path("/old/database/a.nii.gz")
    with pytest.raises(ValueError, match="traversal"):
        resolver.locate("/old/data/../secret.nii.gz")
    with pytest.raises(ValueError, match="unique"):
        ReadOnlyPathResolver([{"source_prefix": "/old/data", "target_prefix": str(tmp_path)}] * 2)
