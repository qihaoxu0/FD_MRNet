#!/usr/bin/env python
"""Generate layout-only artifacts. Outputs are technically blocked from submission mode."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="placeholders/SIMULATED_PLACEHOLDER_NOT_FOR_SUBMISSION")
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    # Values are layout-only projections anchored to the internally conflicting legacy tables.
    rows = [
        ["BraTS2021", "T1", 2, "Trilinear", 35.8, 0.955],
        ["BraTS2021", "T1", 2, "FD-MRNet", 39.0, 0.989],
        ["BraTS2021", "T1", 4, "Trilinear", 30.8, 0.889],
        ["BraTS2021", "T1", 4, "FD-MRNet", 36.8, 0.989],
        ["BraTS2021", "T2", 2, "Trilinear", 30.4, 0.940],
        ["BraTS2021", "T2", 2, "FD-MRNet", 34.0, 0.981],
        ["BraTS2021", "T2", 4, "Trilinear", 28.9, 0.880],
        ["BraTS2021", "T2", 4, "FD-MRNet", 33.0, 0.976],
    ]
    frame = pd.DataFrame(rows, columns=["dataset", "modality", "scale", "method", "psnr", "ssim"])
    frame.insert(0, "warning", "SIMULATED PLACEHOLDER - NOT FOR SUBMISSION")
    frame.to_csv(out / "SIMULATED_main_table.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, group in frame.groupby("method"):
        subset = group[group.modality == "T1"]
        ax.plot(subset.scale, subset.psnr, marker="o", label=method)
    ax.set(xlabel="Through-plane scale", ylabel="PSNR (dB)", xticks=[2, 4])
    ax.legend(); ax.grid(alpha=.2)
    fig.text(.5, .5, "SIMULATED PLACEHOLDER\nNOT FOR SUBMISSION", ha="center", va="center",
             fontsize=24, color="red", alpha=.23, rotation=25)
    fig.tight_layout(); fig.savefig(out / "SIMULATED_psnr_layout.png", dpi=200); plt.close(fig)


if __name__ == "__main__": main()

