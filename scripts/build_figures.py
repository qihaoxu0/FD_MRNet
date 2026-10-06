#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def reject_placeholder(path: Path) -> None:
    upper = str(path).upper()
    if any(token in upper for token in ("SIMULATED", "PLACEHOLDER", "NOT_FOR_SUBMISSION")):
        raise RuntimeError(f"Real-results figure builder rejected placeholder input: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="main_results_long.csv from build_tables.py")
    parser.add_argument("--run-manifest", help="Optional generated_configs/run_manifest.csv for training curves")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summary_path = Path(args.summary)
    reject_placeholder(summary_path)
    summary = pd.read_csv(summary_path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    main = summary[
        (summary.group.isin(["main", "baselines"]))
        & (summary.region == "brain")
        & (summary.metric == "psnr")
    ].copy()
    if not main.empty:
        grid = sns.relplot(
            data=main,
            x="scale",
            y="subject_mean",
            hue="method",
            style="method",
            col="dataset",
            row="modality",
            kind="line",
            marker="o",
            facet_kws={"sharey": False},
            height=3.0,
            aspect=1.25,
        )
        grid.set_axis_labels("Through-plane scale", "Full-volume PSNR (dB)")
        grid.set(xticks=[2, 4])
        grid.figure.savefig(out / "main_psnr_comparison.png", dpi=300, bbox_inches="tight")
        grid.figure.savefig(out / "main_psnr_comparison.pdf", bbox_inches="tight")
        plt.close(grid.figure)

    ablation = summary[
        (summary.group.isin(["main", "ablations"]))
        & (summary.dataset.str.lower() == "brats2021")
        & (summary.modality.str.upper() == "T1")
        & (summary.scale == 4)
        & (summary.region == "brain")
        & (summary.metric == "psnr")
    ].sort_values("subject_mean")
    if not ablation.empty:
        fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(ablation))))
        ax.barh(ablation.method, ablation.subject_mean, xerr=ablation.subject_sd, capsize=2)
        ax.set(xlabel="Full-volume PSNR (dB)", ylabel="", title="BraTS2021 T1 ×4 component ablation")
        fig.tight_layout()
        fig.savefig(out / "ablation_psnr.png", dpi=300, bbox_inches="tight")
        fig.savefig(out / "ablation_psnr.pdf", bbox_inches="tight")
        plt.close(fig)

    if args.run_manifest:
        manifest_path = Path(args.run_manifest)
        reject_placeholder(manifest_path)
        manifest = pd.read_csv(manifest_path)
        curves = []
        for row in manifest[manifest.group == "main"].to_dict("records"):
            path = Path(row["output_dir"]) / "epoch_metrics.csv"
            if not path.is_file():
                continue
            frame = pd.read_csv(path)
            frame["condition"] = f"{row['dataset']} {str(row['modality']).upper()} ×{row['scale']}"
            frame["seed"] = int(row["seed"])
            curves.append(frame)
        if curves:
            curves = pd.concat(curves, ignore_index=True)
            grid = sns.relplot(
                data=curves,
                x="epoch",
                y="val_psnr",
                hue="seed",
                col="condition",
                col_wrap=2,
                kind="line",
                estimator=None,
                units="seed",
                height=2.8,
                aspect=1.35,
                facet_kws={"sharey": False},
            )
            grid.set_axis_labels("Epoch", "Fixed-patch validation PSNR (diagnostic only)")
            grid.figure.savefig(out / "training_convergence_unsmoothed.png", dpi=300, bbox_inches="tight")
            grid.figure.savefig(out / "training_convergence_unsmoothed.pdf", bbox_inches="tight")
            plt.close(grid.figure)


if __name__ == "__main__":
    main()
