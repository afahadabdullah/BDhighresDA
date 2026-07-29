#!/bin/bash
#SBATCH --job-name=bdhires-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate bdhires
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python scripts/train.py --config configs/train_h100.yaml
