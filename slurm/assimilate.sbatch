#!/bin/bash
#SBATCH --job-name=bdhires-da
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=2020-2025
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate bdhires
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
Y=$SLURM_ARRAY_TASK_ID
python scripts/assimilate.py --config configs/da.yaml --ckpt runs/prior_v100/final.pt \
    --start "${Y}-01-01" --end "${Y}-12-31" --mode analysis \
    --out "data/processed/bdhires_analysis_${Y}.nc"
