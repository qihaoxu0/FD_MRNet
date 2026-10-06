#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use LABEL=/path/to/subject.npz")
    label, path = value.split("=", 1)
    return label, Path(path)


def slice2d(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "axial":
        return volume[index]
    if plane == "coronal":
        return volume[:, index, :]
    return volume[:, :, index]


def choose_index(mask: np.ndarray, plane: str) -> int:
    axis = {"axial": 0, "coronal": 1, "sagittal": 2}[plane]
    reduce_axes = tuple(value for value in range(3) if value != axis)
    counts = mask.sum(axis=reduce_axes)
    return int(np.argmax(counts)) if counts.max() else mask.shape[axis] // 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=parse_input, required=True, help="LABEL=subject.npz; repeat per method")
    parser.add_argument("--plane", choices=("axial", "coronal", "sagittal"), default="axial")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    loaded = [(label, np.load(path, allow_pickle=False)) for label, path in args.input]
    target = loaded[0][1]["target"]
    tumor = loaded[0][1]["tumor_mask"].astype(bool)
    index = choose_index(tumor, args.plane)
    images = [("Trilinear", loaded[0][1]["coarse"])]
    for label, data in loaded:
        if data["target"].shape != target.shape or not np.allclose(data["target"], target, atol=1e-6):
            raise RuntimeError(f"Target mismatch for {label}; qualitative comparison is not paired")
        images.append((label, data["prediction"]))
    images.append(("HR target", target))
    fig, axes = plt.subplots(2, len(images), figsize=(3 * len(images), 6), squeeze=False)
    for column, (label, volume) in enumerate(images):
        image = slice2d(volume, args.plane, index)
        reference = slice2d(target, args.plane, index)
        mask2d = slice2d(tumor, args.plane, index)
        axes[0, column].imshow(image, cmap="gray", vmin=0, vmax=1)
        if mask2d.any():
            axes[0, column].contour(mask2d, levels=[0.5], colors="yellow", linewidths=0.6)
        axes[0, column].set_title(label)
        axes[1, column].imshow(np.abs(image - reference), cmap="magma", vmin=0, vmax=0.25)
        axes[1, column].set_title("Absolute error")
        for axis in axes[:, column]:
            axis.axis("off")
    fig.suptitle(f"Paired full-volume comparison; {args.plane} index {index}")
    fig.tight_layout()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
