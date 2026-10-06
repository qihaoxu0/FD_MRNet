"""Technical statistics fixtures; no fixture output supports a manuscript claim."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_statistics import paired_statistics, require_matching_long


def technical_long(subjects=("patient_a", "patient_b")):
    rows = []
    for split in ("val", "test"):
        for regime in ("shared", "alternative"):
            for seed in (2025, 2026, 2027):
                for subject in subjects:
                    for method in ("fdmrnet", "trilinear"):
                        rows.append({"group": "main", "dataset": "BraTS2021", "split": split,
                            "training_regime": regime, "modality": "T1", "scale": 4,
                            "metric": "psnr", "region": "brain", "seed": seed,
                            "subject": subject, "method": method, "evidence_class": "diagnostic",
                            "value": 30. + (1. if method == "fdmrnet" else 0.)})
    return pd.DataFrame(rows)


def test_split_and_training_regime_are_separate_inferential_conditions():
    result = paired_statistics(technical_long())
    assert len(result) == 4
    assert {(row["split"], row["training_regime"]) for row in result} == {
        ("val", "shared"), ("val", "alternative"), ("test", "shared"), ("test", "alternative")}
    assert all(row["n_subjects"] == 2 for row in result)


@pytest.mark.parametrize("change", ["unpaired_subject", "unpaired_seed", "duplicate", "nan", "missing_split"])
def test_pairing_and_primary_metric_failures_stop_statistics(change):
    frame = technical_long()
    if change == "unpaired_subject":
        frame = frame[~((frame.method == "trilinear") & (frame.subject == "patient_b"))]
    elif change == "unpaired_seed":
        frame.loc[frame.method == "trilinear", "seed"] += 1
    elif change == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif change == "nan":
        frame.loc[0, "value"] = np.nan
    else:
        frame = frame.drop(columns="split")
    with pytest.raises(RuntimeError):
        paired_statistics(frame)


def test_single_patient_has_actual_means_but_no_ci_or_pvalue():
    result = paired_statistics(technical_long(subjects=("patient_a",)))
    assert len(result) == 4
    for row in result:
        assert row["fdmrnet_mean"] == 31. and row["delta_mean"] == 1.
        assert row["delta_ci95"] == row["p_raw"] == row["p_holm"] == "N/A"


def test_prespecified_greater_hypothesis_does_not_flip_with_negative_result():
    frame = technical_long()
    frame.loc[frame.method == "fdmrnet", "value"] = 29.
    result = paired_statistics(frame)
    assert all(row["delta_mean"] == -1. for row in result)
    assert all(row["hypothesis"] == "prespecified one-sided greater" for row in result)
    assert all(row["p_raw"] >= .5 for row in result)


def test_changed_long_csv_cannot_replace_verified_registry_data():
    frame = technical_long()
    require_matching_long(frame.copy(), frame)
    altered = frame.copy()
    altered.loc[0, "value"] += .01
    with pytest.raises(RuntimeError, match="differs from verified"):
        require_matching_long(altered, frame)
