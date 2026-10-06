#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import torch

from fdmrnet.config import load_config
from fdmrnet.models import build_model


def parameters(model) -> int:
    return sum(value.numel() for value in model.parameters())


class TensorOutput(torch.nn.Module):
    def __init__(self, model, target_shape):
        super().__init__()
        self.model = model
        self.target_shape = target_shape

    def forward(self, value):
        return self.model(value, self.target_shape)["pred"]


def flops(model, value, target_shape) -> int:
    from fvcore.nn import FlopCountAnalysis

    return int(FlopCountAnalysis(TensorOutput(model, target_shape), value).total())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--channels", nargs="+", type=int, default=[64, 72, 80, 88, 92, 93, 96, 104, 112])
    parser.add_argument("--blocks", nargs="+", type=int, default=[6, 7, 8, 10, 12, 14, 16])
    parser.add_argument("--with-flops", action="store_true")
    parser.add_argument("--profile-shape-dhw", nargs=3, type=int, default=[16, 32, 32])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = load_config(args.target_config)
    target = build_model(config["model"])
    target_parameters = parameters(target)
    target_shape = tuple(args.profile_shape_dhw)
    scale = int(config["degradation"]["scale"])
    value = torch.rand(1, 1, target_shape[0] // scale, target_shape[1], target_shape[2])
    target_flops = flops(target.eval(), value, target_shape) if args.with_flops else None
    candidates = []
    for channels in args.channels:
        for blocks in args.blocks:
            cfg = {"name": "matched_residual3d", "channels": channels, "num_blocks": blocks}
            model = build_model(cfg)
            count = parameters(model)
            parameter_error = abs(count - target_parameters) / target_parameters
            candidate_flops = flops(model.eval(), value, target_shape) if args.with_flops else None
            flop_error = abs(candidate_flops - target_flops) / target_flops if args.with_flops else None
            score = max(parameter_error, flop_error) if args.with_flops else parameter_error
            candidates.append({
                "model": cfg,
                "parameters": count,
                "parameter_relative_error": parameter_error,
                "flops": candidate_flops,
                "flop_relative_error": flop_error,
                "matching_score": score,
            })
    candidates.sort(key=lambda item: item["matching_score"])
    report = {
        "target_parameters": target_parameters,
        "target_flops": target_flops,
        "flop_convention": f"fvcore on HR profile shape {target_shape} with protocol LR depth" if args.with_flops else None,
        "best": candidates[0],
        "top_five": candidates[:5],
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
