#!/bin/bash
#SBATCH --job-name=fdmr-main
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm/logs/%A_%a.out
#SBATCH --error=slurm/logs/%A_%a.err

set -euo pipefail
mkdir -p slurm/logs
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" slurm/main_configs.txt)
echo "config=${CONFIG}"
torchrun --standalone --nproc_per_node=4 scripts/train.py --config "$CONFIG"

