from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Full checkpoints may be loaded with map_location="cuda". The default CPU
    # generator still requires a CPU ByteTensor, while CUDA generators also
    # accept their saved states from CPU.
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_metric: float,
    config: dict,
    manifest_hashes: dict,
    training_state: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "config_sha256": config.get("_config_sha256"),
        "manifest_hashes": manifest_hashes,
        "rng": _rng_state(),
        "training_state": training_state,
    }
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, restore_rng=True, map_location="cpu") -> dict:
    # Full training checkpoints are trusted local artifacts and intentionally require weights_only=False.
    payload = torch.load(path, map_location=map_location, weights_only=False)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        _restore_rng(payload.get("rng"))
    return payload
