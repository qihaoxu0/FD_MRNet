import math

import torch

from fdmrnet.metrics import metric_bundle


def test_identical_volume_metrics():
    target = torch.rand(1, 1, 8, 8, 8)
    mask = torch.ones_like(target, dtype=torch.bool)
    values = metric_bundle(target, target, mask, 1.0)
    assert values["psnr"] >= 119
    assert abs(values["ssim"] - 1.0) < 1e-5
    assert values["nmse"] == 0
    assert values["hfen"] == 0


def test_empty_roi_returns_nan_and_zero_valid_count():
    target = torch.rand(1, 1, 8, 8, 8)
    values = metric_bundle(target, target, torch.zeros_like(target, dtype=torch.bool), 1.0)
    assert values["valid_n"] == 0
    assert all(math.isnan(values[key]) for key in ("psnr", "ssim", "nmse", "hfen"))
