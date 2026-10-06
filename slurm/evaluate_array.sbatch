#!/bin/bash
#SBATCH --job-name=fdmr-eval
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/logs/eval_%A_%a.out
#SBATCH --error=slurm/logs/eval_%A_%a.err

set -euo pipefail
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" slurm/main_configs.txt)
OUTPUT=$(python -c "from fdmrnet.config import load_config; print(load_config('$CONFIG')['experiment']['output_dir'])")
python scripts/evaluate_full_volume.py --config "$CONFIG" --checkpoint "$OUTPUT/best.pt" --split val --tag final_val
python scripts/evaluate_full_volume.py --config "$CONFIG" --checkpoint "$OUTPUT/best.pt" --split test --tag final_test

