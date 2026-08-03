#!/bin/bash
#SBATCH --job-name=bdhires-osse-footprint
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-83%6
#SBATCH --output=logs/%x-%A_%a.out

# Seven matched arms x 12 JJA year-month chunks. The common 120x120 window is
# exactly divisible by factors 2, 4 and 10, so resolution is the only satellite
# likelihood difference among the 0.1, 0.2 and 0.5 degree arms.
set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"
PYTHON_BIN="$ENV_PREFIX/bin/python"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"

CKPT="${OSSE_CKPT:-runs/prior_h100_cpc/best.pt}"
CONFIG="${OSSE_CONFIG:-configs/da.yaml}"
ROOT="${OSSE_FOOTPRINT_ROOT:-data/processed/osse_footprint_ablation_50_2021_2024}"
MEMBERS="${OSSE_FOOTPRINT_MEMBERS:-16}"
WITHHOLD="${OSSE_FOOTPRINT_WITHHOLD:-0.2}"
SEED="${OSSE_FOOTPRINT_SEED:-7}"

ARM_INDEX=$((SLURM_ARRAY_TASK_ID / 12))
GROUP_INDEX=$((SLURM_ARRAY_TASK_ID % 12))
YEAR=$((2021 + GROUP_INDEX / 3))
MONTH=$((6 + GROUP_INDEX % 3))

case "$ARM_INDEX" in
    0) LABEL="gauges_exact_50_common"; MODE="gauges"; FACTOR=2 ;;
    1) LABEL="satellite_exact_50_f2"; MODE="satellite"; FACTOR=2 ;;
    2) LABEL="satellite_exact_50_f4"; MODE="satellite"; FACTOR=4 ;;
    3) LABEL="satellite_exact_50_f10"; MODE="satellite"; FACTOR=10 ;;
    4) LABEL="simultaneous_exact_50_f2"; MODE="combined"; FACTOR=2 ;;
    5) LABEL="simultaneous_exact_50_f4"; MODE="combined"; FACTOR=4 ;;
    6) LABEL="simultaneous_exact_50_f10"; MODE="combined"; FACTOR=10 ;;
    *) echo "ERROR: unknown arm index $ARM_INDEX"; exit 2 ;;
esac

CHUNK="${YEAR}_$(printf '%02d' "$MONTH")"
OUT="$ROOT/chunks/$LABEL/$CHUNK"
mkdir -p "$OUT"

echo "$LABEL / $CHUNK: factor=$FACTOR, common crop=(4,124,8,128), members=$MEMBERS"
srun --ntasks=1 "$PYTHON_BIN" -u scripts/10_osse.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --start "$YEAR-06-01" \
    --end "$YEAR-08-31" \
    --month "$MONTH" \
    --days 31 \
    --members "$MEMBERS" \
    --networks 50 \
    --station-layout spread \
    --holdout-layout spread \
    --withhold "$WITHHOLD" \
    --seed "$SEED" \
    --obs-error exact \
    --observation-mode "$MODE" \
    --satellite-factor "$FACTOR" \
    --satellite-crop 4 124 8 128 \
    --satellite-stride 1 \
    --dump "$OUT/ensemble.npz" \
    --dump-network 50 \
    --dump-obs-error exact \
    --out-figure "$OUT/osse_overview.png" \
    --out-report "$OUT/osse_report.json"

test -s "$OUT/ensemble.npz"
echo "completed $LABEL / $CHUNK"
