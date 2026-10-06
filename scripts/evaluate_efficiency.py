#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from fdmrnet.config import load_config, sha256_file
from fdmrnet.engine.checkpoint import load_checkpoint
from fdmrnet.engine.sliding_window import sliding_window_predict
from fdmrnet.models import build_model


class TensorOutput(torch.nn.Module):
    def __init__(self, model, target_shape):
        super().__init__()
        self.model = model
        self.target_shape = target_shape

    def forward(self, x):
        return self.model(x, self.target_shape)["pred"]


def count_flops(model, patch: torch.Tensor, target_shape) -> tuple[int | None, str]:
    try:
        from fvcore.nn import FlopCountAnalysis

        analysis = FlopCountAnalysis(TensorOutput(model, target_shape), patch)
        total = int(analysis.total())
        unsupported = {key: int(value) for key, value in analysis.unsupported_ops().items()}
        return total, f"fvcore operators; unsupported={unsupported}"
    except Exception as exc:  # profiler support differs by architecture/PyTorch build
        return None, f"FLOP profiling unavailable: {type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--volume-shape-dhw", nargs=3, type=int, default=[155, 240, 240])
    parser.add_argument("--submission-mode", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.submission_mode and not args.checkpoint:
        raise RuntimeError("Submission-mode efficiency audit requires the evaluated checkpoint")
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device).eval()
    if args.checkpoint:
        payload = load_checkpoint(args.checkpoint, model, restore_rng=False, map_location=device)
        if payload.get("config_sha256") != cfg.get("_config_sha256"):
            raise RuntimeError("Efficiency evaluation refused: checkpoint/config hash mismatch")
    patch_shape = tuple(cfg["evaluation"]["patch_size_dhw"])
    patch = torch.rand((1, 1, *patch_shape), device=device)
    amp = cfg["evaluation"].get("amp", True) and device.type == "cuda"
    flops, flops_note = count_flops(model, patch, patch_shape)
    with torch.inference_mode():
        for _ in range(args.warmup):
            with torch.autocast(device_type=device.type, enabled=amp):
                model(patch, patch_shape)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        volume = torch.rand((1, 1, *tuple(args.volume_shape_dhw)), device=device)
        times, patch_count = [], None
        for _ in range(args.repeats):
            start = time.perf_counter()
            _, audit = sliding_window_predict(
                model,
                volume,
                patch_shape,
                tuple(cfg["evaluation"]["overlap_dhw"]),
                cfg["evaluation"].get("gaussian_sigma_scale", 0.125),
                amp,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            patch_count = audit["patches"]
    gpu = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    report = {
        "method": cfg["model"]["name"],
        "device": str(device),
        "gpu_name": gpu.name if gpu else None,
        "gpu_memory_bytes": gpu.total_memory if gpu else None,
        "torch_version": torch.__version__,
        "cudnn_version": torch.backends.cudnn.version(),
        "precision": "AMP" if amp else "FP32",
        "batch_size": 1,
        "preprocessing_scope": "excluded for every method; timing starts from normalized coarse HR-grid volume",
        "volume_shape_dhw": list(args.volume_shape_dhw),
        "patch_shape_dhw": list(patch_shape),
        "overlap_dhw": list(cfg["evaluation"]["overlap_dhw"]),
        "patches_per_volume": patch_count,
        "warmup_patch_iterations": args.warmup,
        "full_volume_repeats": args.repeats,
        "full_volume_seconds_mean": statistics.mean(times),
        "full_volume_seconds_sd": statistics.stdev(times) if len(times) > 1 else 0.0,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "patch_flops": flops,
        "estimated_volume_flops": flops * patch_count if flops is not None else None,
        "flops_note": flops_note,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
        "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint else None,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
