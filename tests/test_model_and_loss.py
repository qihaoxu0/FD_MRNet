import torch
import pytest

from fdmrnet.losses import CompositeFDMRNetLoss
from fdmrnet.models import build_model


def tiny_config():
    return {
        "name": "fdmrnet",
        "channels": 8,
        "num_blocks": 2,
        "heads": 2,
        "window_dhw": [2, 2, 2],
        "frequency_sigma": 1.0,
        "max_coarse_disp": 1.0,
        "max_residual_disp": 0.5,
        "gradient_checkpointing": False,
        "use_frequency": True,
        "use_attention": True,
        "use_cross_gate": True,
        "use_adaptive_fusion": True,
        "use_hierarchical_offsets": True,
        "use_residual_propagation": True,
        "use_global_residual": True,
    }


def test_forward_shapes_and_finite_backward():
    model = build_model(tiny_config())
    lr = torch.rand(1, 1, 4, 8, 8)
    hr = torch.rand(1, 1, 8, 8, 8)
    outputs = model(lr, hr.shape[-3:])
    assert outputs["pred"].shape == hr.shape
    assert outputs["d_low"].shape == (1, 2, 3, 8, 8, 8)
    criterion = CompositeFDMRNetLoss()
    loss, parts = criterion(
        outputs,
        {
            "hr": hr,
            "brain_mask": torch.ones_like(hr, dtype=torch.bool),
            "field_target": torch.zeros(1, 3, 8, 8, 8),
        },
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_baseline_factory_ignores_inherited_fdmrnet_keys():
    cfg = tiny_config() | {"name": "edsr3d", "channels": 8, "num_blocks": 2}
    model = build_model(cfg)
    output = model(torch.rand(1, 1, 4, 8, 8), (8, 8, 8))["pred"]
    assert output.shape == (1, 1, 8, 8, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP is required")
def test_jacobian_loss_amp_path_is_fp32_finite_and_matches_fp32():
    device = torch.device("cuda")
    shape = (8, 8, 8)
    z = torch.arange(shape[0], device=device, dtype=torch.float32).view(1, 1, shape[0], 1, 1)
    folding = torch.zeros((1, 3, *shape), device=device, dtype=torch.float32)
    folding[:, 0:1] = -2.0 * z

    def run(dtype, amp):
        field = folding.to(dtype).detach().requires_grad_(True)
        zeros = torch.zeros((1, 1, *shape), device=device, dtype=torch.float32)
        outputs = {
            "pred": zeros, "pred_low": zeros, "pred_high": zeros, "coarse": zeros,
            "d_low": field.unsqueeze(1), "d_high": torch.zeros_like(field).unsqueeze(1),
            "d_composite": field,
        }
        batch = {
            "hr": zeros, "brain_mask": torch.ones_like(zeros, dtype=torch.bool),
            "field_target": torch.zeros_like(field),
        }
        criterion = CompositeFDMRNetLoss(
            lambda_rec=0.0, lambda_freq=0.0, lambda_align=0.0,
            lambda_field=0.0, lambda_smooth=0.0, lambda_jacobian=1.0,
        ).to(device)
        with torch.autocast("cuda", enabled=amp):
            loss, parts = criterion(outputs, batch)
        loss.backward()
        return loss.detach(), parts["jacobian"].detach(), field.grad.detach()

    amp_loss, amp_jac, amp_grad = run(torch.float16, True)
    fp32_loss, fp32_jac, fp32_grad = run(torch.float32, False)
    assert amp_loss.dtype == torch.float32
    assert amp_jac.dtype == torch.float32
    assert torch.isfinite(amp_loss) and torch.isfinite(amp_grad).all()
    assert torch.count_nonzero(amp_grad).item() > 0
    assert torch.allclose(amp_jac, fp32_jac, rtol=2e-3, atol=2e-4)
    assert torch.allclose(amp_grad.float(), fp32_grad, rtol=2e-2, atol=2e-3)
