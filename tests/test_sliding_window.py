import torch

from fdmrnet.engine.sliding_window import sliding_window_predict


class IdentityModel(torch.nn.Module):
    def forward(self, x, target_shape_dhw):
        assert tuple(x.shape[-3:]) == tuple(target_shape_dhw)
        return {"pred": x}


def test_gaussian_overlap_reconstructs_identity():
    volume = torch.rand(1, 1, 10, 12, 14)
    pred, audit = sliding_window_predict(IdentityModel(), volume, (8, 8, 8), (4, 4, 4), amp=False)
    assert torch.allclose(pred, volume, atol=1e-6)
    assert audit["patches"] > 1
    assert audit["coverage_min"] > 0
