import torch

from fdmrnet.degradation import ThroughPlaneDegrader


def test_through_plane_shape_and_axis_only():
    hr = torch.rand(2, 1, 16, 12, 10)
    degrader = ThroughPlaneDegrader(scale=4, sigma_vox=1.6985)
    lr, target, meta = degrader.degrade(hr)
    assert lr.shape == (2, 1, 4, 12, 10)
    assert target.shape == hr.shape
    assert meta.original_depth == 16
    assert meta.lr_depth == 4


def test_non_divisible_depth_is_padded_and_audited():
    hr = torch.rand(1, 1, 15, 8, 8)
    lr, target, meta = ThroughPlaneDegrader(scale=4, sigma_vox=0).degrade(hr)
    assert target.shape[-3] == 16
    assert lr.shape[-3] == 4
    assert meta.original_depth == 15
    assert meta.padded_depth == 16


def test_no_blur_center_decimation_is_exact():
    hr = torch.arange(16.0).view(1, 1, 16, 1, 1)
    lr, _, _ = ThroughPlaneDegrader(scale=2, sigma_vox=0, sample_offset="center").degrade(hr)
    assert torch.equal(lr.flatten(), torch.arange(1.0, 16.0, 2.0))
