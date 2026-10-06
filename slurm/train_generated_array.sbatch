#!/bin/bash
# Submit with a site-appropriate concurrency cap, e.g. --array=0-$((N-1))%4
#SBATCH --job-name=fdmr-formal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm/logs/%A_%a.out
#SBATCH --error=slurm/logs/%A_%a.err

set -euo pipefail
LIST=${1:-generated_configs/all_configs.txt}
mkdir -p slurm/logs
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
test -n "$CONFIG"
echo "config=${CONFIG}"
torchrun --standalone --nproc_per_node=4 scripts/train.py --config "$CONFIG"
