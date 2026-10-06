from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubjectRecord:
    subject: str
    t1: str
    t2: str
    seg: str


def _find_one(folder: Path, suffixes: tuple[str, ...], required: bool = True) -> str:
    matches = [p for p in folder.iterdir() if p.is_file() and p.name.lower().endswith(suffixes)]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous files in {folder}: {[p.name for p in matches]}")
    if not matches:
        if required:
            raise FileNotFoundError(f"Missing {suffixes} in {folder}")
        return ""
    return str(matches[0].resolve())


def _record_from_folder(folder: Path, require_seg: bool) -> SubjectRecord | None:
    try:
        t1 = _find_one(folder, ("_t1.nii.gz", "-t1n.nii.gz"))
        t2 = _find_one(folder, ("_t2.nii.gz", "-t2w.nii.gz"))
        seg = _find_one(folder, ("_seg.nii.gz", "-seg.nii.gz"), required=require_seg)
    except FileNotFoundError:
        return None
    return SubjectRecord(folder.name, t1, t2, seg)


def discover_brats_subjects(
    root: str | Path,
    require_seg: bool = True,
    corrections_root: str | Path | None = None,
    *,
    external_eval_without_seg: bool = False,
) -> list[SubjectRecord]:
    if external_eval_without_seg:
        require_seg = False
    elif not require_seg:
        raise ValueError(
            "Missing segmentation is permitted only with "
            "external_eval_without_seg=True"
        )
    root = Path(root)
    records: dict[str, SubjectRecord] = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        record = _record_from_folder(folder, require_seg)
        if record is not None:
            records[record.subject] = record
    if corrections_root:
        corrections_root = Path(corrections_root)
        for folder in sorted(path for path in corrections_root.iterdir() if path.is_dir()):
            record = _record_from_folder(folder, require_seg)
            if record is None:
                raise RuntimeError(f"Incomplete correction case: {folder}")
            if record.subject not in records:
                raise RuntimeError(f"Correction subject is absent from base dataset: {record.subject}")
            records[record.subject] = record
    if not records:
        raise RuntimeError(f"No complete BraTS subjects discovered under {root}")
    return [records[key] for key in sorted(records)]


def write_manifest(records: list[SubjectRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "t1", "t2", "seg"])
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def read_manifest(path: str | Path) -> list[SubjectRecord]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return [SubjectRecord(**row) for row in csv.DictReader(f)]
