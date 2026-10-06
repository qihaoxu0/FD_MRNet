# FD-MRNet

A PyTorch implementation for single-input, same-contrast 3D through-plane MRI super-resolution. T1w and T2w are separate reconstruction tasks; supported scale factors are x2 and x4.

## Project layout

- `fdmrnet/`: models, data loading, degradation, losses, geometry, metrics and evaluation.
- `configs/`: model, baseline, ablation and training configuration examples.
- `scripts/`: dataset auditing, training, inference and result utilities.
- `tests/`: implementation checks.
- `slurm/`: scheduler examples.
- `environment.yml`, `requirements.txt`, `pyproject.toml`: dependency and package definitions.
- `CODE_MANIFEST.sha256`: checksums for the exported project files.

## Installation

```bash
conda env create -f environment.yml
conda activate fdmrnet-rr
pip install -e .
```

## Getting started

Edit dataset and manifest paths in the configuration files before running:

```bash
pytest -q
python scripts/preflight.py --config configs/main/brats2021_t1_x2.yaml
python scripts/train.py --config configs/main/brats2021_t1_x2.yaml
```

## Implementation

The default FD-MRNet uses 48 feature channels and six frequency-aware feature blocks. It combines low/high-frequency decomposition, local window attention, hierarchical feature resampling, adaptive fusion and residual reconstruction.

The internal `swinir3d` implementation is a compact 3D window-attention baseline. It is not an official full SwinIR implementation. HF error evaluation uses a normalized discrete 3D Laplacian; it does not use a Laplacian-of-Gaussian filter.

This is a reimplementation snapshot. Model definitions and experiment outputs should be matched by configuration and source identity before quantitative comparisons.

## Scope

This repository distributes source code and configuration examples only. Historical documents, checkpoints and pilot outputs are archived separately. This code release does not assert reproduction of historical quantitative tables.
