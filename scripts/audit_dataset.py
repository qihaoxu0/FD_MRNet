#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from fdmrnet.data.discovery import discover_brats_subjects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--corrections-root")
    parser.add_argument("--out", required=True)
    parser.add_argument("--external-eval-without-seg", action="store_true")
    args = parser.parse_args()
    records = discover_brats_subjects(
        args.root,
        corrections_root=args.corrections_root,
        external_eval_without_seg=args.external_eval_without_seg,
    )
    cases, shapes, affine_mismatch, unreadable = [], {}, [], []
    canonical_hash_owners: dict[tuple[str, str], list[str]] = {}

    def file_hash(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def canonical_voxel_hash(image: nib.spatialimages.SpatialImage) -> str:
        canonical = nib.as_closest_canonical(image)
        array = np.asarray(canonical.dataobj, dtype=np.float32)
        digest = hashlib.sha256()
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    for record in records:
        row = {"subject": record.subject}
        reference_affine = None
        for modality in ("t1", "t2", "seg"):
            path = getattr(record, modality)
            if not path:
                row[modality] = None
                continue
            try:
                image = nib.load(path)
                image.get_fdata(dtype=np.float32)
            except Exception as exc:
                unreadable.append({"subject": record.subject, "modality": modality, "error": repr(exc)})
                continue
            row[f"{modality}_shape"] = list(image.shape)
            row[f"{modality}_spacing"] = [float(v) for v in image.header.get_zooms()[:3]]
            row[f"{modality}_orientation"] = list(nib.aff2axcodes(image.affine))
            row[f"{modality}_affine"] = np.asarray(image.affine).tolist()
            row[f"{modality}_file_sha256"] = file_hash(path)
            row[f"{modality}_canonical_voxel_sha256"] = canonical_voxel_hash(image)
            shapes[str(tuple(image.shape))] = shapes.get(str(tuple(image.shape)), 0) + 1
            if reference_affine is None:
                reference_affine = image.affine
            elif not np.allclose(reference_affine, image.affine, atol=1e-4):
                affine_mismatch.append({"subject": record.subject, "modality": modality})
        if row.get("t1_canonical_voxel_sha256") and row.get("t2_canonical_voxel_sha256"):
            key = (row["t1_canonical_voxel_sha256"], row["t2_canonical_voxel_sha256"])
            canonical_hash_owners.setdefault(key, []).append(record.subject)
        cases.append(row)
    within_dataset_duplicates = [
        {"subjects": subjects, "t1_canonical_voxel_sha256": key[0], "t2_canonical_voxel_sha256": key[1]}
        for key, subjects in canonical_hash_owners.items()
        if len(subjects) > 1
    ]
    report = {
        "root": str(Path(args.root).resolve()),
        "corrections_root": str(Path(args.corrections_root).resolve()) if args.corrections_root else None,
        "subjects": len(records),
        "subject_range": [records[0].subject, records[-1].subject],
        "shape_counts": shapes,
        "affine_mismatches": affine_mismatch,
        "unreadable_nifti": unreadable,
        "within_dataset_canonical_pair_duplicates": within_dataset_duplicates,
        "segmentation_policy": (
            "N/A: segmentation unavailable; external evaluation only"
            if args.external_eval_without_seg
            else "required"
        ),
        "cases": cases,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("subjects", "subject_range", "shape_counts", "affine_mismatches", "unreadable_nifti", "within_dataset_canonical_pair_duplicates", "segmentation_policy")}, indent=2))


if __name__ == "__main__":
    main()
