from pathlib import Path

import yaml

from fdmrnet.config import load_config


def test_missing_seg_requires_explicit_external_mode(tmp_path):
    from fdmrnet.data.discovery import discover_brats_subjects

    try:
        discover_brats_subjects(tmp_path, require_seg=False)
    except ValueError as exc:
        assert "external_eval_without_seg=True" in str(exc)
    else:
        raise AssertionError("implicit missing-seg mode was accepted")


def test_dataset_missing_seg_requires_explicit_external_mode(tmp_path):
    import csv

    from fdmrnet.data.dataset import _BaseBraTS

    manifest = tmp_path / "external.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["subject", "t1", "t2", "seg"])
        writer.writeheader()
        writer.writerow({"subject": "case", "t1": "t1.nii.gz", "t2": "t2.nii.gz", "seg": ""})
    try:
        _BaseBraTS(manifest, "t1")
    except ValueError as exc:
        assert "external_eval_without_seg" in str(exc)
    else:
        raise AssertionError("dataset silently accepted missing segmentation")
    dataset = _BaseBraTS(manifest, "t1", external_eval_without_seg=True)
    assert len(dataset) == 1


def test_all_repository_configs_load():
    root = Path(__file__).parents[1]
    for folder in ("main", "ablations", "baselines"):
        for path in sorted((root / "configs" / folder).glob("*.yaml")):
            cfg = load_config(path)
            assert cfg["degradation"]["scale"] in (2, 4)
            assert cfg["data"]["modality"] in ("t1", "t2")


def test_recursive_inheritance_is_deep_merged(tmp_path):
    base = {
        "experiment": {"id": "x", "seed": 1, "output_dir": "out"},
        "data": {"modality": "t1", "patch_size_dhw": [8, 8, 8]},
        "degradation": {"axis": "depth", "scale": 2},
        "model": {"name": "fdmrnet", "channels": 8},
        "loss": {},
        "training": {},
        "evaluation": {},
    }
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    (tmp_path / "child.yaml").write_text("base: base.yaml\nmodel:\n  channels: 12\n", encoding="utf-8")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["model"]["channels"] == 12
    assert cfg["model"]["name"] == "fdmrnet"
