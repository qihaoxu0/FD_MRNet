import torch

from fdmrnet.geometry import invert_displacement, jacobian_determinant, resize_displacement, upsample_displacement, warp_volume
from fdmrnet.metrics.geometry import geometry_metrics


def test_zero_field_is_identity_and_has_unit_jacobian():
    x = torch.rand(1, 1, 8, 9, 10)
    field = torch.zeros(1, 3, 8, 9, 10)
    assert torch.allclose(warp_volume(x, field), x, atol=1e-6)
    assert torch.allclose(jacobian_determinant(field), torch.ones(1, 7, 8, 9))
    metrics = geometry_metrics(field, field)
    assert metrics["field_epe_vox"] == 0
    assert metrics["folding_percent"] == 0


def test_displacement_upsampling_rescales_voxel_units():
    field = torch.ones(1, 3, 2, 4, 4)
    up = upsample_displacement(field, (4, 8, 8))
    assert torch.allclose(up, torch.full_like(up, 2.0))


def test_resize_round_trip_and_inverse_field():
    field = torch.ones(1, 3, 4, 8, 8) * 0.1
    low = resize_displacement(field, (2, 4, 4))
    restored = resize_displacement(low, field.shape[-3:])
    assert torch.allclose(restored, field, atol=1e-6)
    inverse = invert_displacement(field, iterations=5)
    assert torch.allclose(inverse, -field, atol=1e-5)
