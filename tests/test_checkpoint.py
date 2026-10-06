import random

import numpy as np
import torch

from fdmrnet.engine.checkpoint import load_checkpoint, save_checkpoint


def test_full_checkpoint_roundtrip_restores_training_and_rng(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward(); optimizer.step(); scheduler.step()
    expected_model = {key: value.clone() for key, value in model.state_dict().items()}
    path = tmp_path / "full.pt"
    save_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        epoch=10, best_metric=1.0, config={"_config_sha256": "cfg"}, manifest_hashes={"train": "split"},
    )
    expected_rng = (random.random(), float(np.random.rand()), torch.rand(1).item())
    for parameter in model.parameters():
        parameter.data.zero_()
    payload = load_checkpoint(path, model, optimizer, scheduler, scaler, restore_rng=True)
    restored_rng = (random.random(), float(np.random.rand()), torch.rand(1).item())
    assert payload["epoch"] == 10
    assert payload["manifest_hashes"] == {"train": "split"}
    assert all(torch.equal(model.state_dict()[key], value) for key, value in expected_model.items())
    assert scheduler.state_dict()["T_max"] == 100
    assert restored_rng == expected_rng
