#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from fdmrnet.data.discovery import read_manifest


def voxel_hash(path: str) -> str:
    image = nib.as_closest_canonical(nib.load(path))
    array = np.asarray(image.dataobj, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    # Canonical voxel payload ignores filename, storage orientation and metadata.
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def manifest_index(path: str, full_voxel_hash: bool) -> dict[str, dict]:
    output = {}
    for record in read_manifest(path):
        output[record.subject] = {
            "subject": record.subject,
            "t1_hash": voxel_hash(record.t1) if full_voxel_hash else None,
            "t2_hash": voxel_hash(record.t2) if full_voxel_hash else None,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-a", required=True, help="A split manifest or all-subject manifest")
    parser.add_argument("--dataset-b", required=True)
    parser.add_argument("--full-voxel-hash", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    a, b = manifest_index(args.dataset_a, args.full_voxel_hash), manifest_index(args.dataset_b, args.full_voxel_hash)
    id_overlap = sorted(set(a) & set(b))
    voxel_overlap = []
    if args.full_voxel_hash:
        reverse: dict[tuple[str, str], list[str]] = {}
        for item in b.values():
            reverse.setdefault((item["t1_hash"], item["t2_hash"]), []).append(item["subject"])
        for item in a.values():
            for other in reverse.get((item["t1_hash"], item["t2_hash"]), []):
                voxel_overlap.append({"dataset_a_subject": item["subject"], "dataset_b_subject": other})
    report = {
        "dataset_a_manifest": str(Path(args.dataset_a).resolve()),
        "dataset_b_manifest": str(Path(args.dataset_b).resolve()),
        "dataset_a_subjects": len(a),
        "dataset_b_subjects": len(b),
        "subject_id_overlap": id_overlap,
        "voxel_identical_overlap": voxel_overlap,
        "full_voxel_hash_performed": args.full_voxel_hash,
        "independent_external_cohort": not id_overlap and (not args.full_voxel_hash or not voxel_overlap),
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("dataset_a_subjects", "dataset_b_subjects", "subject_id_overlap", "voxel_identical_overlap", "independent_external_cohort")}, indent=2))


if __name__ == "__main__":
    main()
