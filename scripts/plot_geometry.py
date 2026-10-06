#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def jacobian(field: np.ndarray) -> np.ndarray:
    # field is 3,D,H,W with component order D,H,W.
    gradients = [[np.gradient(field[i], axis=j) for j in range(3)] for i in range(3)]
    matrix = np.empty(field.shape[1:] + (3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            matrix[..., i, j] = gradients[i][j] + (1.0 if i == j else 0.0)
    return np.linalg.det(matrix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="geometry_examples/case_000.npz")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = np.load(args.input, allow_pickle=False)
    pred, target = data["predicted_field"], data["target_field"]
    pred_mag = np.linalg.norm(pred, axis=0)
    target_mag = np.linalg.norm(target, axis=0)
    epe = np.linalg.norm(pred - target, axis=0)
    det = jacobian(pred)
    index = pred.shape[1] // 2
    panels = [
        ("Target displacement magnitude", target_mag[index], "viridis"),
        ("Predicted displacement magnitude", pred_mag[index], "viridis"),
        ("Endpoint error", epe[index], "magma"),
        ("Predicted Jacobian determinant", det[index], "coolwarm"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for axis, (title, image, cmap) in zip(axes, panels):
        shown = axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
