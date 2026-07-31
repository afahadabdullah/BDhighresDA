#!/bin/bash
#SBATCH --job-name=bdhires-bases
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Score candidate residual bases against CHIRPS. No GPU needed.
set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"
PYTHON_BIN="$ENV_PREFIX/bin/python"
export PYTHONNOUSERSITE=1
unset PYTHONHOME; unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: not found: $PYTHON_BIN"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
[[ "$(uname -m)" == "aarch64" ]] || { echo "ERROR: needs an aarch64 Grace node"; exit 1; }

STATS="${BASES_STATS:-data/processed/stats_v3.json}"
CPC_DIR="${BASES_CPC_DIR:-data/raw/cpc}"
DAYS="${BASES_DAYS:-400}"
MONTH="${BASES_MONTH:-0}"
START="${BASES_START:-2000-01-01}"
END="${BASES_END:-2018-12-31}"

echo "Statistics: $STATS   CPC: $CPC_DIR   Days: $DAYS   Month: $MONTH"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/13_compare_bases.py \
    --stats "$STATS" \
    --cpc-dir "$CPC_DIR" \
    --days "$DAYS" \
    --month "$MONTH" \
    --start "$START" \
    --end "$END" \
    --out-figure "${BASES_FIGURE:-data/processed/base_comparison.png}" \
    --out-report "${BASES_REPORT:-data/processed/base_comparison.json}"
