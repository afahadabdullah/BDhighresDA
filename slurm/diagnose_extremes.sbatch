#!/bin/bash
#SBATCH --job-name=bdhires-extremes
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Attribute the unphysical member extremes seen in the v5 epoch-150 panel.
# See scripts/09_diagnose_extremes.py for what the three hypotheses are.

set -euo pipefail

module purge 2>/dev/null || true

export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"
PYTHON_BIN="$ENV_PREFIX/bin/python"

export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || {
    echo "ERROR: ARM training Python not found: $PYTHON_BIN"
    exit 1
}

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: This job must run on an aarch64 Grace node."
    exit 1
}

CKPT="${EXTREME_CKPT:-runs/prior_h100_v5/best.pt}"
CONFIG="${EXTREME_CONFIG:-configs/da.yaml}"
DAYS="${EXTREME_DAYS:-40}"
MEMBERS="${EXTREME_MEMBERS:-8}"
MONTH="${EXTREME_MONTH:-7}"
FIGURE="${EXTREME_FIGURE:-data/processed/extreme_diagnosis.png}"
REPORT="${EXTREME_REPORT:-data/processed/extreme_diagnosis.json}"

[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }

echo "Checkpoint: $CKPT"
echo "Config: $CONFIG"
echo "Days: $DAYS   Members: $MEMBERS   Month: $MONTH"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/09_diagnose_extremes.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --days "$DAYS" \
    --members "$MEMBERS" \
    --month "$MONTH" \
    --out-figure "$FIGURE" \
    --out-report "$REPORT"
