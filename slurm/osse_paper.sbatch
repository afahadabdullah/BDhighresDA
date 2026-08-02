#!/bin/bash
#SBATCH --job-name=bdhires-osse-paper
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-6%2
#SBATCH --output=logs/%x-%A_%a.out

# Matched paper OSSE arms. CHIRPS is the 0.05-degree nature truth and supplies
# pseudo-gauges plus exact nested 0.1-degree pseudo-IMERG footprints. Tasks use
# identical dates, seeds and spatial holdout except for explicit density arms.

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"
PYTHON_BIN="$ENV_PREFIX/bin/python"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: not found: $PYTHON_BIN"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: This job must run on an aarch64 Grace node."
    exit 1
}

CKPT="${OSSE_CKPT:-runs/prior_h100_cpc/best.pt}"
CONFIG="${OSSE_CONFIG:-configs/da.yaml}"
DAYS="${OSSE_PAPER_DAYS:-48}"
MEMBERS="${OSSE_PAPER_MEMBERS:-16}"
START="${OSSE_PAPER_START:-2021-01-01}"
END="${OSSE_PAPER_END:-2024-12-31}"
WITHHOLD="${OSSE_PAPER_WITHHOLD:-0.2}"
SEED="${OSSE_PAPER_SEED:-7}"
ROOT="${OSSE_PAPER_ROOT:-data/processed/osse_paper}"

case "$SLURM_ARRAY_TASK_ID" in
    0) LABEL="gauges_realistic_40"; MODE="gauges"; ERROR="realistic"; NETWORK="40"; STRIDE="1"; CONTROL="0" ;;
    1) LABEL="satellite_realistic_40"; MODE="satellite"; ERROR="realistic"; NETWORK="40"; STRIDE="3"; CONTROL="1" ;;
    2) LABEL="simultaneous_realistic_40"; MODE="combined"; ERROR="realistic"; NETWORK="40"; STRIDE="3"; CONTROL="1" ;;
    3) LABEL="simultaneous_perfect_40"; MODE="combined"; ERROR="perfect"; NETWORK="40"; STRIDE="3"; CONTROL="0" ;;
    4) LABEL="simultaneous_dense_uncorrected_40"; MODE="combined"; ERROR="realistic"; NETWORK="40"; STRIDE="1"; CONTROL="0" ;;
    5) LABEL="simultaneous_realistic_20"; MODE="combined"; ERROR="realistic"; NETWORK="20"; STRIDE="3"; CONTROL="1" ;;
    6) LABEL="simultaneous_realistic_80"; MODE="combined"; ERROR="realistic"; NETWORK="80"; STRIDE="3"; CONTROL="1" ;;
    *) echo "ERROR: unknown task $SLURM_ARRAY_TASK_ID"; exit 2 ;;
esac

OUT="$ROOT/$LABEL"
mkdir -p "$OUT"
EXTRA=(--satellite-stride "$STRIDE")
[[ "$CONTROL" == "1" ]] && EXTRA+=(--satellite-correlation-control)

echo "Paper OSSE arm: $LABEL"
echo "mode=$MODE error=$ERROR network=$NETWORK stride=$STRIDE correlation_control=$CONTROL"
echo "window=$START..$END days=$DAYS members=$MEMBERS seed=$SEED"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/10_osse.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --start "$START" \
    --end "$END" \
    --month 0 \
    --days "$DAYS" \
    --members "$MEMBERS" \
    --networks "$NETWORK" \
    --station-layout spread \
    --withhold "$WITHHOLD" \
    --seed "$SEED" \
    --obs-error "$ERROR" \
    --observation-mode "$MODE" \
    --dump "$OUT/ensemble.npz" \
    --dump-network "$NETWORK" \
    --dump-obs-error "$ERROR" \
    --out-figure "$OUT/osse_overview.png" \
    --out-report "$OUT/osse_report.json" \
    "${EXTRA[@]}"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/11_da_diagnostics.py \
    --dump "$OUT/ensemble.npz" \
    --out-figure "$OUT/da_diagnostics.png" \
    --out-report "$OUT/da_diagnostics.json"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/14_osse_summary.py \
    --dump "$OUT/ensemble.npz" \
    --cases 4 \
    --out-reconstruction "$OUT/reconstruction.png" \
    --out-matrix "$OUT/scale_matrix.png" \
    --out-report "$OUT/scale_summary.json"

# Scale-separated downscaling evaluation.  Emits the claim-A / claim-B report,
# the curve NPZ that 24_osse_paper_suite.py replots, and georeferenced maps.
srun --ntasks=1 "$PYTHON_BIN" -u scripts/22_evaluate_osse_downscaling.py \
    --dump "$OUT/ensemble.npz" \
    --label "$LABEL" \
    --minimum-valid-fraction 1.0 \
    --out-figure "$OUT/downscaling_diagnostics.png" \
    --out-report "$OUT/downscaling.json" \
    --out-curve-data "$OUT/downscaling_curves.npz" \
    --out-spatial-data "$OUT/spatial_fields.nc"

# Spatial impact is most useful for the primary simultaneous arm; avoid making
# seven nearly redundant large figures.
if [[ "$SLURM_ARRAY_TASK_ID" == "2" ]]; then
    srun --ntasks=1 "$PYTHON_BIN" -u scripts/12_da_impact_maps.py \
        --dump "$OUT/ensemble.npz" \
        --cases 4 \
        --out-cases "$OUT/impact_cases.png" \
        --out-aggregate "$OUT/impact_aggregate.png" \
        --out-report "$OUT/impact.json"
fi

echo "completed $LABEL"
