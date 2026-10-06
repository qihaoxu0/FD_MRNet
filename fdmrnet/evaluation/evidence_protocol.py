"""Immutable, separately frozen evaluation of archived training artifacts.

This module never rewrites a training configuration, manifest, or checkpoint.
Path relocation changes only the read location and requires an independently
hashed image inventory. Diagnostic outputs are explicitly ineligible for tables.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import re
import shutil
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fdmrnet.config import canonical_hash, load_config, sha256_file
from fdmrnet.data.dataset import BraTSFullVolumeDataset
from fdmrnet.data.discovery import read_manifest
from fdmrnet.degradation import ThroughPlaneDegrader
from fdmrnet.engine.sliding_window import sliding_window_predict
from fdmrnet.metrics import metric_bundle
from fdmrnet.models import build_model


NA = "N/A"
METRICS = ("psnr", "ssim", "nmse", "hfen", "valid_n")


def _json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _path(reference: str, base: Path) -> Path:
    path = Path(reference)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _sha(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError(f"{name} must be an explicit lowercase SHA256")
    return value


def _checked_file(reference, base: Path, name: str):
    path = _path(reference["path"], base)
    expected = _sha(reference["sha256"], name)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{name} hash mismatch: {path}")
    return path


class ReadOnlyPathResolver:
    """Boundary-aware prefix relocation with identity checks before image reads."""

    def __init__(self, mappings: list[dict], inventory: dict | None = None):
        self.mappings = []
        seen = set()
        for mapping in mappings:
            source = mapping["source_prefix"].replace("\\", "/").rstrip("/")
            target = Path(mapping["target_prefix"])
            if not source or not target.is_absolute() or source in seen:
                raise ValueError("Path mappings need unique nonempty sources and absolute targets")
            seen.add(source)
            self.mappings.append((source, target.resolve()))
        self.mappings.sort(key=lambda pair: len(pair[0]), reverse=True)
        self.inventory = inventory

    def locate(self, original: str) -> Path:
        normalized = original.replace("\\", "/")
        for source, target in self.mappings:
            if normalized == source or normalized.startswith(source + "/"):
                relative = normalized[len(source):].lstrip("/")
                if ".." in Path(relative).parts:
                    raise ValueError("Parent traversal is not permitted in relocated paths")
                resolved = (target / relative).resolve()
                if not resolved.is_relative_to(target):
                    raise ValueError("Relocated path escaped the target directory")
                return resolved
        candidate = Path(original)
        if not candidate.is_absolute():
            raise ValueError(f"Unmapped relative source image: {original}")
        return candidate.resolve()

    def verify(self, original: str) -> tuple[Path, dict]:
        path = self.locate(original)
        if self.inventory is None or original not in self.inventory.get("files", {}):
            raise RuntimeError(f"Image lacks a frozen identity inventory: {original}")
        expected = self.inventory["files"][original]
        if sha256_file(path) != _sha(expected.get("sha256"), "image identity"):
            raise RuntimeError(f"Image identity hash mismatch: {original}")
        image = nib.load(str(path))
        if len(image.shape) != 3 or list(image.shape) != expected["shape_xyz"]:
            raise RuntimeError(f"Image geometry shape mismatch: {original}")
        affine = np.asarray(expected["affine"], dtype=float)
        if affine.shape != (4, 4) or not np.isfinite(image.affine).all():
            raise RuntimeError(f"Invalid image affine: {original}")
        if not np.allclose(image.affine, affine, rtol=0, atol=1e-6):
            raise RuntimeError(f"Image geometry affine mismatch: {original}")
        return path, {"original_path": original, "resolved_path": str(path), **expected}


def build_file_inventory(manifest, path_map, modality, subject_ids=None):
    """Freeze current byte identities, without claiming identity was checked at training."""
    resolver = ReadOnlyPathResolver(_json(path_map)["mappings"])
    records = _selected_records(manifest, subject_ids)
    files = {}
    for record in records:
        images = (record.t1, record.t2) if modality == "both" else (getattr(record, modality.lower()),)
        for original in (*images, record.seg):
            if not original or original in files:
                continue
            path = resolver.locate(original)
            image = nib.load(str(path))
            if len(image.shape) != 3 or not np.isfinite(image.affine).all():
                raise RuntimeError(f"Invalid NIfTI geometry: {path}")
            files[original] = {
                "sha256": sha256_file(path), "shape_xyz": list(image.shape),
                "affine": image.affine.tolist(), "resolved_path_at_freeze": str(path),
            }
    return {
        "schema_version": 1, "manifest_sha256": sha256_file(manifest),
        "path_map_sha256": sha256_file(path_map), "modality": modality.lower(),
        "subject_ids": [record.subject for record in records],
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "identity_scope": "current files; not independently proven bound at training",
        "files": files,
    }


def _selected_records(manifest, subject_ids):
    records = read_manifest(manifest)
    roster = [record.subject for record in records]
    if not roster or len(roster) != len(set(roster)):
        raise ValueError("Manifest must have nonempty, unique subject identifiers")
    if subject_ids is None:
        return records
    if not isinstance(subject_ids, list) or not subject_ids or len(set(subject_ids)) != len(subject_ids):
        raise ValueError("subject_ids must be null (all) or a nonempty unique list")
    if set(subject_ids) - set(roster):
        raise ValueError("Selected subjects are absent from the original manifest")
    selected = set(subject_ids)
    return [record for record in records if record.subject in selected]


class RelocatedFullVolumeDataset(BraTSFullVolumeDataset):
    def __init__(self, manifest, modality, degrader, resolver, records, clip_z, allow_missing_seg):
        super().__init__(manifest, modality, degrader, clip_z,
                         external_eval_without_seg=allow_missing_seg)
        self.identity_audit = []
        relocated = []
        for record in records:
            image_path, image_audit = resolver.verify(getattr(record, modality))
            item = {"subject": record.subject, "image": image_audit}
            changes = {modality: str(image_path)}
            if record.seg:
                seg_path, seg_audit = resolver.verify(record.seg)
                if seg_audit["shape_xyz"] != image_audit["shape_xyz"] or not np.allclose(
                    seg_audit["affine"], image_audit["affine"], rtol=0, atol=1e-4
                ):
                    raise RuntimeError(f"Segmentation geometry differs for {record.subject}")
                changes["seg"] = str(seg_path)
                item["segmentation"] = seg_audit
            relocated.append(replace(record, **changes))
            self.identity_audit.append(item)
        self.records = relocated


def _source_code_hashes():
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "fdmrnet").rglob("*.py"))
    files.append(root / "scripts" / "evaluate_evidence_protocol.py")
    return {str(path): sha256_file(path) for path in files}


def _snapshot_source_files(source_files, output, tag):
    root = Path(__file__).resolve().parents[2]
    destination = output / f"{tag}_source_snapshot"
    destination.mkdir(exist_ok=False)
    snapshots = {}
    for original, expected in source_files.items():
        source = Path(original)
        target = destination / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if sha256_file(source) != expected:
            raise RuntimeError("Executed source changed before its immutable snapshot")
        with source.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        if sha256_file(target) != expected:
            raise RuntimeError("Source snapshot does not match executed source bytes")
        snapshots[str(target)] = expected
    return snapshots


def _bind_method(protocol, cfg):
    factory = str(cfg["model"]["name"]).lower()
    method = protocol.get("method", factory)
    variant_hash = canonical_hash({"model": cfg["model"], "loss": cfg["loss"]})
    group = protocol.get("evaluation_group", "main")
    if group == "ablations":
        binding = protocol.get("method_binding", {})
        if binding.get("factory_model") != factory or binding.get("config_variant_sha256") != variant_hash or binding.get("variant_name") != method:
            raise RuntimeError("Ablation method requires an explicit source-config-backed variant binding")
        factories = {"fdmrnet", "edsr3d", "rdn3d", "swinir3d", "spectralsr3d", "matched_residual3d", "trilinear"}
        if method in factories and method != factory:
            raise RuntimeError("An ablation cannot be renamed as a different baseline factory")
    elif group in {"main", "baselines"}:
        if method != factory:
            raise RuntimeError("Main/baseline method name differs from the checkpoint's model factory")
    else:
        raise ValueError("evaluation_group must be main, baselines, or ablations")
    return method, factory, group, variant_hash


def _training_budget(cfg, payload, protocol, base):
    observed = {key: payload[key] for key in ("epoch", "step", "micro_step", "micro_steps", "global_step", "optimizer_steps") if key in payload}
    steps = []
    for state in payload.get("optimizer", {}).get("state", {}).values():
        if "step" in state:
            value = state["step"]
            value = float(value.item()) if torch.is_tensor(value) else float(value)
            if not math.isfinite(value) or value < 0 or value != int(value):
                raise RuntimeError("Saved optimizer contains an invalid parameter step counter")
            steps.append(int(value))
    observed["optimizer_parameter_step_counters"] = {
        "count": len(steps), "minimum": min(steps) if steps else None,
        "maximum": max(steps) if steps else None, "unique": sorted(set(steps)),
        "interpretation": "saved per-parameter Adam step counters; not inferred from micro-batches",
    }
    observed["scheduler_state"] = {key: payload.get("scheduler", {}).get(key) for key in ("last_epoch", "T_max", "base_lrs")}
    observed["optimizer_parameter_groups"] = [
        {key: group.get(key) for key in ("lr", "initial_lr", "weight_decay", "betas")}
        for group in payload.get("optimizer", {}).get("param_groups", [])
    ]
    declared = {
        **copy.deepcopy(cfg["training"]), "train_samples_per_epoch": cfg["data"].get("train_samples_per_epoch"),
        "patch_size_dhw": cfg["data"]["patch_size_dhw"], "augmentation": cfg["data"].get("augmentation", {}),
        "optimizer": "AdamW", "scheduler": {"name": "CosineAnnealingLR", "T_max": cfg["training"]["epochs"]},
    }
    result = {"declared_source_config": declared, "checkpoint_observations": observed,
              "verification": "not_verified_diagnostic; nominal config is not proof of executed overrides"}
    reference = protocol.get("training_budget_evidence")
    if not reference:
        if protocol["evidence_class"] == "formal":
            raise RuntimeError("Formal evidence requires independently hashed training_budget_evidence; nominal epoch/config alone is insufficient")
        return result, None
    path = _checked_file(reference, base, "training budget evidence")
    evidence = _json(path)
    if evidence.get("checkpoint_sha256") != protocol["checkpoint"]["sha256"] or evidence.get("source_config_sha256") != cfg["_config_sha256"]:
        raise RuntimeError("Training budget evidence does not bind this checkpoint and source config")
    numeric = ("epochs_completed", "optimizer_updates", "micro_batches", "effective_global_batch_size",
               "world_size", "batch_size_per_gpu", "gradient_accumulation", "scheduler_steps")
    if any(not isinstance(evidence.get(key), int) or isinstance(evidence[key], bool) or evidence[key] <= 0 for key in numeric):
        raise RuntimeError("Training budget evidence has missing or nonpositive actual counters")
    training = cfg["training"]
    if (evidence["epochs_completed"] != int(payload.get("epoch", -1)) or
            evidence["epochs_completed"] != int(training["epochs"]) or
            evidence["batch_size_per_gpu"] != int(training["batch_size_per_gpu"]) or
            evidence["gradient_accumulation"] != int(training.get("gradient_accumulation", 1)) or
            evidence["effective_global_batch_size"] != int(training["target_global_batch_size"]) or
            evidence["world_size"] * evidence["batch_size_per_gpu"] * evidence["gradient_accumulation"] != evidence["effective_global_batch_size"]):
        raise RuntimeError("Recorded actual epochs or batch/accumulation budget differs from source config")
    if evidence["training_manifest_sha256"] != payload.get("manifest_hashes", {}).get("train"):
        raise RuntimeError("Training budget evidence has a different training manifest")
    if evidence.get("optimizer") != "AdamW" or evidence.get("scheduler") != declared["scheduler"]:
        raise RuntimeError("Recorded optimizer or scheduler differs from the native training contract")
    for key in ("learning_rate", "weight_decay"):
        if key not in evidence or not math.isclose(float(evidence[key]), float(training[key]), rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeError(f"Recorded actual {key} differs from source config")
    groups = payload.get("optimizer", {}).get("param_groups", [])
    base_lrs = payload.get("scheduler", {}).get("base_lrs", [])
    if not groups or not base_lrs or len(base_lrs) != len(groups):
        raise RuntimeError("Saved optimizer/scheduler lacks verified initial learning-rate groups")
    for group, base_lr in zip(groups, base_lrs):
        if (not math.isclose(float(base_lr), float(training["learning_rate"]), rel_tol=1e-9, abs_tol=1e-12) or
                group.get("initial_lr") is None or not math.isclose(float(group["initial_lr"]), float(base_lr), rel_tol=1e-9, abs_tol=1e-12) or
                group.get("weight_decay") is None or not math.isclose(float(group["weight_decay"]), float(training["weight_decay"]), rel_tol=1e-9, abs_tol=1e-12)):
            raise RuntimeError("Saved optimizer initial LR or weight decay differs from source config")
    if not steps or max(steps) != evidence["optimizer_updates"]:
        raise RuntimeError("Recorded optimizer updates do not match the maximum saved parameter step counter")
    if payload.get("scheduler", {}).get("last_epoch") != evidence["scheduler_steps"] or payload.get("scheduler", {}).get("T_max") != int(training["epochs"]):
        raise RuntimeError("Recorded scheduler steps do not match saved scheduler state")
    if evidence["scheduler_steps"] != evidence["epochs_completed"] or evidence["micro_batches"] < evidence["optimizer_updates"]:
        raise RuntimeError("Actual scheduler or micro-batch counters are inconsistent")
    logs = evidence.get("source_log_files")
    if not isinstance(logs, dict) or not logs:
        raise RuntimeError("Actual training counters require independently hashed source logs")
    for log_path, expected in logs.items():
        if sha256_file(_path(log_path, path.parent)) != _sha(expected, "training source log"):
            raise RuntimeError("Training budget source log hash mismatch")
    result.update({"verification": "verified_runtime_budget_binding", "evidence": evidence,
                   "evidence_path": str(path), "evidence_sha256": sha256_file(path)})
    return result, path


def _validate_fresh_overlap(report, protocol, base, manifest, inventory, payload):
    # Count-only historical reports are retained as history, never silently upgraded.
    if (report.get("schema_version") != 1 or report.get("status") != "completed" or
            report.get("full_voxel_hash_performed") is not True or
            report.get("independent_external_cohort") is not True or
            report.get("subject_id_overlap") != [] or report.get("voxel_identical_overlap") != []):
        raise RuntimeError("Formal external evaluation requires a fresh completed full-voxel overlap audit; legacy counts do not qualify")
    if report.get("canonical_voxel_hash_algorithm") != "nibabel.as_closest_canonical; float32; shape+contiguous voxel bytes":
        raise RuntimeError("Fresh voxel audit must identify the exact canonical voxel hash algorithm")
    required = ("training_cohort_manifest", "training_cohort_inventory", "training_cohort_path_map")
    if any(key not in protocol for key in required):
        raise RuntimeError("Fresh external audit requires explicit complete training-cohort manifest/inventory/path-map bindings")
    training_manifest = _checked_file(protocol["training_cohort_manifest"], base, "training cohort manifest")
    training_inventory_path = _checked_file(protocol["training_cohort_inventory"], base, "training cohort inventory")
    training_path_map = _checked_file(protocol["training_cohort_path_map"], base, "training cohort path map")
    training_inventory = _json(training_inventory_path)
    if (report.get("dataset_a_manifest_sha256") != sha256_file(training_manifest) or
            report.get("dataset_b_manifest_sha256") != sha256_file(manifest) or
            report.get("dataset_a_file_inventory_sha256") != sha256_file(training_inventory_path) or
            report.get("dataset_b_file_inventory_sha256") != protocol["file_inventory"]["sha256"]):
        raise RuntimeError("Fresh voxel audit is not bound to the evaluated raw manifests and image inventories")
    if (training_inventory.get("manifest_sha256") != sha256_file(training_manifest) or
            training_inventory.get("path_map_sha256") != sha256_file(training_path_map) or
            training_inventory.get("modality") != "both" or inventory.get("modality") != "both"):
        raise RuntimeError("Fresh voxel audit requires frozen identities for both T1 and T2 in both cohorts")
    all_training = read_manifest(training_manifest)
    external = read_manifest(manifest)
    splits = protocol.get("training_split_manifests", {})
    if set(splits) != {"train", "val", "test"}:
        raise RuntimeError("Complete BraTS2021 cohort must be bound to its original train/val/test manifest union")
    union = {}
    for split, reference in splits.items():
        split_path = _checked_file(reference, base, f"training cohort {split} manifest")
        bound = payload.get("manifest_hashes", {}).get(split)
        if bound is not None and bound != sha256_file(split_path):
            raise RuntimeError("Training-cohort split differs from its checkpoint-bound manifest")
        for record in read_manifest(split_path):
            if record.subject in union:
                raise RuntimeError("Training-cohort splits have overlapping subject IDs")
            union[record.subject] = record
    if {record.subject: record for record in all_training} != union:
        raise RuntimeError("Audited complete training cohort differs from the original split union")
    for key, records in (("dataset_a", all_training), ("dataset_b", external)):
        roster = sorted(record.subject for record in records)
        if report.get(f"{key}_subjects") != len(records) or report.get(f"{key}_subject_ids") != roster:
            raise RuntimeError("Fresh voxel audit subject roster/count is not the frozen cohort")
    if set(union) & {record.subject for record in external}:
        raise RuntimeError("External cohort has subject-ID overlap with the complete training cohort")
    training_resolver = ReadOnlyPathResolver(_json(training_path_map)["mappings"], training_inventory)
    external_resolver = ReadOnlyPathResolver(_json(_checked_file(protocol["path_map"], base, "external path map"))["mappings"], inventory)
    for records, resolver in ((all_training, training_resolver), (external, external_resolver)):
        for record in records:
            resolver.verify(record.t1)
            resolver.verify(record.t2)


def _freeze_policy(protocol, cfg, payload, manifest_sha):
    split, evidence = protocol["split"], protocol["evidence_class"]
    if split not in {"val", "test", "external"} or evidence not in {"diagnostic", "formal"}:
        raise ValueError("Unsupported split or evidence_class")
    freeze = protocol["freeze"]
    frozen_at = datetime.fromisoformat(freeze["frozen_at"].replace("Z", "+00:00"))
    if frozen_at.tzinfo is None or frozen_at > datetime.now(timezone.utc):
        raise ValueError("Freeze time must be a real timezone-aware time, not a future time")
    if not freeze.get("checkpoint_selection_rule") or freeze.get("no_test_tuning") is not True:
        raise ValueError("An explicit checkpoint-selection rule and no_test_tuning=true are required")
    checkpoint_manifest = payload.get("manifest_hashes", {}).get(split)
    binding = "bound_at_training" if checkpoint_manifest is not None else "not_bound_at_training"
    if freeze.get("manifest_binding") != binding:
        raise RuntimeError("Protocol misstates whether the split was bound by the training checkpoint")
    if checkpoint_manifest is not None and checkpoint_manifest != manifest_sha:
        raise RuntimeError("Original split manifest differs from the checkpoint-bound hash")
    if binding == "bound_at_training":
        key = f"{split}_manifest"
        if key not in cfg["data"]:
            raise RuntimeError("Checkpoint-bound split lacks its source-config manifest reference")
    if split == "external":
        if protocol["dataset"] != "BraTS2023" or cfg["data"]["dataset"] != "BraTS2021":
            raise ValueError("External cohort must be explicitly BraTS2023, separate from BraTS2021 training")
    elif protocol["dataset"] != cfg["data"]["dataset"]:
        raise ValueError("Internal evaluation dataset differs from the source configuration")
    if evidence == "formal":
        if protocol.get("fixture") is not False or split == "val":
            raise ValueError("Fixtures and validation diagnostics are not formal submission results")
        if protocol.get("subject_ids") is not None:
            raise ValueError("Formal evaluation must include the entire frozen original manifest")
        if freeze.get("checkpoint_selection_rule") != "final_epoch":
            raise ValueError("Formal version 1 accepts only the prespecified final_epoch selection rule")
        epochs = int(cfg["training"]["epochs"])
        if int(payload.get("epoch", -1)) != epochs:
            raise RuntimeError("Formal evaluation requires completion of the source-config epoch budget")
        for field in ("step", "micro_step", "micro_steps", "optimizer_steps", "global_step"):
            if field in payload and int(payload[field]) <= 500:
                raise RuntimeError("500-step engineering candidates cannot supply formal evidence")
        if protocol.get("evaluation"):
            raise ValueError("Formal evaluation cannot override source-config inference settings")
        if split in {"test", "external"}:
            if freeze.get("never_used_for_training_tuning_or_selection") is not True:
                raise ValueError("Held-out policy must explicitly exclude training, tuning and selection")
            if binding == "not_bound_at_training" and not freeze.get("late_binding_reason"):
                raise ValueError("Late-bound held-out cohort requires an explicit truthful explanation")
        if split == "external":
            overlap = protocol.get("overlap_audit")
            if not overlap:
                raise ValueError("Formal external evaluation requires the frozen voxel-overlap audit")
    return binding


def evaluate_evidence_protocol(protocol_path, output_dir, tag, *, device="cpu", threads=None):
    protocol_path = Path(protocol_path).resolve()
    base, protocol = protocol_path.parent, _json(protocol_path)
    protocol_sha = sha256_file(protocol_path)
    if protocol.get("schema_version") != 1 or not protocol.get("protocol_id"):
        raise ValueError("A named schema_version=1 independent evaluation protocol is required")
    if not isinstance(protocol.get("fixture"), bool):
        raise ValueError("fixture must be explicitly true or false")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag) or tag in {".", ".."}:
        raise ValueError("Output tag must be a simple filename component")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path, audit_path = output / f"{tag}_subject_metrics.csv", output / f"{tag}_audit.json"
    lock_path = output / f"{tag}.lock"
    if csv_path.exists() or audit_path.exists():
        raise RuntimeError("Refusing to overwrite existing evidence outputs")
    # The reservation survives failures to prevent silent replacement of partial runs.
    with lock_path.open("x", encoding="utf-8") as stream:
        json.dump({"protocol_sha256": protocol_sha, "status": "reserved"}, stream)
    cfg_path = _path(protocol["source_config"]["path"], base)
    cfg = load_config(cfg_path)
    config_file_sha = sha256_file(cfg_path)
    if cfg["_config_sha256"] != _sha(protocol["source_config"]["sha256"], "source configuration"):
        raise RuntimeError("Source configuration canonical hash mismatch")
    checkpoint = _checked_file(protocol["checkpoint"], base, "checkpoint")
    manifest = _checked_file(protocol["original_manifest"], base, "original manifest")
    path_map = _checked_file(protocol["path_map"], base, "path map")
    inventory_path = _checked_file(protocol["file_inventory"], base, "file inventory")
    inventory = _json(inventory_path)
    if inventory.get("manifest_sha256") != sha256_file(manifest) or inventory.get("path_map_sha256") != sha256_file(path_map):
        raise RuntimeError("Image inventory was frozen for a different manifest or path map")
    modality = cfg["data"]["modality"].lower()
    if inventory.get("modality") not in {modality, "both"}:
        raise RuntimeError("Inventory modality differs from the model's training modality")
    records = _selected_records(manifest, protocol.get("subject_ids"))
    if inventory.get("subject_ids") != [record.subject for record in records]:
        raise RuntimeError("Inventory subject roster differs from the protocol")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("config_sha256") != cfg["_config_sha256"] or canonical_hash(payload.get("config")) != cfg["_config_sha256"]:
        raise RuntimeError("Checkpoint embedded configuration and source-config hashes differ")
    binding = _freeze_policy(protocol, cfg, payload, sha256_file(manifest))
    method, factory_model, evaluation_group, variant_sha = _bind_method(protocol, cfg)
    training_budget, training_budget_path = _training_budget(cfg, payload, protocol, base)
    overlap_path = None
    if protocol.get("overlap_audit"):
        overlap_path = _checked_file(protocol["overlap_audit"], base, "overlap audit")
        overlap = _json(overlap_path)
        if protocol["evidence_class"] == "formal":
            _validate_fresh_overlap(overlap, protocol, base, manifest, inventory, payload)
    # All immutable inputs are verified before strict model loading and image inference.
    run_device = torch.device(device)
    if run_device.type not in {"cpu", "cuda"}:
        raise ValueError("Only explicit CPU or CUDA inference is supported")
    if threads is not None:
        if threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(threads)
    model = build_model(cfg["model"])
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(run_device).eval()
    evaluation = copy.deepcopy(cfg["evaluation"])
    evaluation.update(protocol.get("evaluation", {}))
    patch, overlap = tuple(evaluation["patch_size_dhw"]), tuple(evaluation["overlap_dhw"])
    if len(patch) != 3 or len(overlap) != 3 or any(p <= 0 or o < 0 or o >= p for p, o in zip(patch, overlap)):
        raise ValueError("Invalid sliding-window patch/overlap")
    dcfg = {key: value for key, value in cfg["degradation"].items() if key != "axis"}
    degrader = ThroughPlaneDegrader(**dcfg)
    allow_missing = protocol["split"] == "external" and protocol.get("external_eval_without_seg") is True
    resolver = ReadOnlyPathResolver(_json(path_map)["mappings"], inventory)
    dataset = RelocatedFullVolumeDataset(manifest, modality, degrader, resolver, records,
                                         cfg["data"].get("clip_z", 5.0), allow_missing)
    source_code = _source_code_hashes()
    rows, case_audits = [], []
    data_range = float(evaluation.get("data_range", 1.0))
    with torch.inference_mode():
        for index, record in enumerate(records):
            item = dataset[index]
            lr, target = item["lr"].unsqueeze(0).to(run_device), item["hr"].unsqueeze(0).to(run_device)
            brain = item["brain_mask"].unsqueeze(0).to(run_device)
            tumor = item["tumor_mask"].unsqueeze(0).to(run_device)
            coarse = degrader.coarse(lr, tuple(target.shape[-3:]))
            if run_device.type == "cuda":
                torch.cuda.synchronize(run_device)
                torch.cuda.reset_peak_memory_stats(run_device)
            started = time.perf_counter()
            pred, window = sliding_window_predict(model, coarse, patch, overlap,
                evaluation.get("gaussian_sigma_scale", 0.125), evaluation.get("amp", True))
            if run_device.type == "cuda":
                torch.cuda.synchronize(run_device)
            seconds = time.perf_counter() - started
            depth = int(item["original_depth"])
            pred, coarse, target, brain, tumor = [value[:, :, :depth] for value in (pred, coarse, target, brain, tumor)]
            if not torch.isfinite(pred).all() or float(window["coverage_min"]) <= 0:
                raise RuntimeError(f"Nonfinite prediction or incomplete coverage: {record.subject}")
            row = {
                "subject": record.subject, "dataset": protocol["dataset"], "split": protocol["split"],
                "modality": modality, "scale": int(cfg["degradation"]["scale"]), "method": method,
                "seed": int(cfg["experiment"]["seed"]), "seconds": seconds,
                "segmentation_available": bool(record.seg), "evidence_class": protocol["evidence_class"],
            }
            for name, prediction in (("baseline", coarse), ("sr", pred)):
                brain_values = metric_bundle(prediction, target, brain, data_range)
                if any(not math.isfinite(brain_values[key]) for key in METRICS):
                    raise RuntimeError(f"Invalid brain metric: {record.subject}")
                row.update({f"{name}_brain_{key}": value for key, value in brain_values.items()})
                if record.seg and tumor.any():
                    values = metric_bundle(prediction, target, tumor, data_range)
                    if any(not math.isfinite(values[key]) for key in METRICS):
                        raise RuntimeError(f"Invalid nonempty tumor metric: {record.subject}")
                else:
                    values = {key: NA for key in METRICS}
                    if record.seg:
                        values["valid_n"] = 0  # Genuine empty-ROI voxel count, not a fabricated metric.
                row.update({f"{name}_tumor_{key}": value for key, value in values.items()})
            row["tumor_metrics"] = "available" if record.seg and tumor.any() else ("N/A: empty tumor ROI" if record.seg else "N/A: segmentation unavailable")
            rows.append(row)
            case_audits.append({"subject": record.subject, "target_shape_dhw": list(target.shape[-3:]),
                "lr_shape_dhw": list(lr.shape[-3:]), "segmentation_available": bool(record.seg),
                "seconds": seconds, "peak_allocated_mib": torch.cuda.max_memory_allocated(run_device) / 2**20 if run_device.type == "cuda" else 0, **window})
            print(json.dumps({"subject": record.subject, "completed": len(rows), "total": len(records)}), flush=True)
    # Check bytes again after inference; concurrent changes invalidate the run.
    for reference, name in (("checkpoint", "checkpoint"), ("original_manifest", "original manifest"),
                            ("path_map", "path map"), ("file_inventory", "file inventory")):
        _checked_file(protocol[reference], base, name)
    if load_config(cfg_path)["_config_sha256"] != cfg["_config_sha256"] or sha256_file(protocol_path) != protocol_sha:
        raise RuntimeError("Source configuration or protocol changed during evaluation")
    for item in dataset.identity_audit:
        for key in ("image", "segmentation"):
            if key in item and sha256_file(item[key]["resolved_path"]) != item[key]["sha256"]:
                raise RuntimeError("Image bytes changed during evaluation")
    if _source_code_hashes() != source_code:
        raise RuntimeError("Source code changed during evaluation")
    if training_budget_path is not None:
        _checked_file(protocol["training_budget_evidence"], base, "training budget evidence")
    snapshot_source = _snapshot_source_files(source_code, output, tag)
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    available = [bool(record.seg) for record in records]
    audit = {
        "status": "completed", "schema_version": 1, "protocol_id": protocol["protocol_id"],
        "evidence_class": protocol["evidence_class"], "fixture": protocol["fixture"], "full_volume": True,
        "split": protocol["split"], "dataset": protocol["dataset"], "training_dataset": cfg["data"]["dataset"],
        "modality": modality, "scale": int(cfg["degradation"]["scale"]), "method": method,
        "factory_model": factory_model, "evaluation_group": evaluation_group,
        "config_variant_sha256": variant_sha,
        "seed": int(cfg["experiment"]["seed"]), "cases": len(rows), "subject_ids": [row["subject"] for row in rows],
        "segmentation_available": all(available) if all(available) or not any(available) else "mixed",
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "source_config_path": str(cfg_path), "source_config_sha256": cfg["_config_sha256"],
        "config_file_sha256": config_file_sha,
        "original_manifest_path": str(manifest), "original_manifest_sha256": sha256_file(manifest),
        "manifest_sha256": sha256_file(manifest), "manifest_binding": binding, "freeze": protocol["freeze"],
        "protocol_path": str(protocol_path), "protocol_sha256": protocol_sha,
        "metrics_path": str(csv_path), "metrics_sha256": sha256_file(csv_path),
        "path_map_path": str(path_map), "path_map_sha256": sha256_file(path_map),
        "file_inventory_path": str(inventory_path), "file_inventory_sha256": sha256_file(inventory_path),
        "source_code_files": snapshot_source, "source_code_sha256": canonical_hash(snapshot_source),
        "executed_source_code_files": source_code,
        "source_code_snapshot_policy": "Immutable byte copies verified against the executed source; live files may change after completion",
        "training_budget": training_budget,
        "checkpoint_training_metadata": {key: payload[key] for key in ("epoch", "step", "micro_step", "micro_steps", "global_step") if key in payload},
        "degradation": cfg["degradation"], "inference": evaluation,
        "normalization": {"scope": "per HR subject nonzero voxels", "clip_z": cfg["data"].get("clip_z", 5.0), "orientation": "DHW=(Z,X,Y)"},
        "runtime": {"device": str(run_device), "torch_version": torch.__version__, "threads": torch.get_num_threads()},
        "metric_protocol": {"unit": "subject-level full3Dvolume", "data_range": data_range,
            "mask": "HR nonzero foreground; tumor label >0 when available", "ssim_window": 7, "ssim_sigma": 1.5,
            "ssim_dimensionality": "3D Gaussian window with zero padding", "hfen": "normalized discrete 3D Laplacian error",
            "prediction_clipping": [0, data_range], "subject_metric_before_cohort_aggregation": True},
        "cases_audit": case_audits, "file_identity_audit": dataset.identity_audit,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    if overlap_path is not None:
        audit["overlap_audit_path"] = str(overlap_path)
        audit["overlap_audit_sha256"] = sha256_file(overlap_path)
    if training_budget_path is not None:
        audit["training_budget_evidence_path"] = str(training_budget_path)
        audit["training_budget_evidence_sha256"] = sha256_file(training_budget_path)
    with audit_path.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    lock_path.unlink()
    return csv_path
