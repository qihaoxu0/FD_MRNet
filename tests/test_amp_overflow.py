import copy
import pytest
import torch

from fdmrnet.engine.amp_step import complete_amp_optimizer_step
from fdmrnet.engine.checkpoint import load_checkpoint, save_checkpoint


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GradScaler is required")
def test_amp_overflow_skips_update_reduces_scale_and_recovers(tmp_path):
    device = torch.device("cuda")
    model = torch.nn.Linear(2, 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scaler = torch.amp.GradScaler("cuda", init_scale=131072.0)
    before = copy.deepcopy(model.state_dict())

    loss = model(torch.ones(1, 2, device=device)).sum()
    scaler.scale(loss).backward()
    next(model.parameters()).grad.view(-1)[0] = torch.inf
    overflow = complete_amp_optimizer_step(
        scaler=scaler, optimizer=optimizer, scheduler=scheduler,
        parameters=model.named_parameters(), gradient_clip=1.0,
    )
    assert overflow["overflow"] and not overflow["optimizer_updated"]
    assert overflow["old_scale"] == 131072.0 and overflow["new_scale"] == 65536.0
    assert scheduler.last_epoch == 0
    assert all(torch.equal(model.state_dict()[key], value) for key, value in before.items())

    optimizer.zero_grad(set_to_none=True)
    finite_loss = model(torch.ones(1, 2, device=device)).sum()
    scaler.scale(finite_loss).backward()
    finite = complete_amp_optimizer_step(
        scaler=scaler, optimizer=optimizer, scheduler=scheduler,
        parameters=model.named_parameters(), gradient_clip=1.0,
    )
    assert not finite["overflow"] and finite["optimizer_updated"]
    assert scheduler.last_epoch == 1
    assert any(not torch.equal(model.state_dict()[key], value) for key, value in before.items())

    runtime = {"weekly_runtime_state": {"successful_optimizer_steps": 1, "skipped_optimizer_steps": 1,
                                        "consecutive_overflows": 0, "overflow_steps": [1]}}
    checkpoint = tmp_path / "amp.pt"
    save_checkpoint(checkpoint, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    epoch=2, best_metric=0.0, config=runtime, manifest_hashes={"train": "hash"})
    payload = load_checkpoint(checkpoint, model, optimizer, scheduler, scaler, restore_rng=True, map_location=device)
    state = payload["config"]["weekly_runtime_state"]
    assert state["successful_optimizer_steps"] == 1
    assert state["skipped_optimizer_steps"] == 1
    assert state["overflow_steps"] == [1]
