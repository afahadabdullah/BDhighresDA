#!/bin/bash
#SBATCH --job-name=bdhires-osse-final
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/%x-%A_%a.out

# Final matched CHIRPS OSSE observation-ablation suite.
#
# CHIRPS is the same-day 0.05-degree nature truth. Exact physical CHIRPS
# values are sampled at actual BMD station coordinates and/or averaged over
# nested 2x2 cells to form exact 0.1-degree pseudo-satellite footprints. No
# random observation error is added. The three arms differ only in which
# observation operators enter the likelihood; checkpoint, dates, seeds,
# members, station split and background conditioning are identical.

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
START="${OSSE_FINAL_START:-2021-01-01}"
END="${OSSE_FINAL_END:-2024-12-31}"
MONTHS="${OSSE_FINAL_MONTHS:-6,7,8}"
DAYS="${OSSE_FINAL_DAYS:-36}"
MEMBERS="${OSSE_FINAL_MEMBERS:-16}"
WITHHOLD="${OSSE_FINAL_WITHHOLD:-0.2}"
SEED="${OSSE_FINAL_SEED:-7}"
ROOT="${OSSE_FINAL_ROOT:-data/processed/osse_final_jja_2021_2024}"
BMD_STATIONS="${OSSE_BMD_STATIONS:-data/bmd/Stations.csv}"
if [[ ! -f "$BMD_STATIONS" && -f data/stations/Stations.csv ]]; then
    BMD_STATIONS="data/stations/Stations.csv"
fi

[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
[[ -f "$BMD_STATIONS" ]] || {
    echo "ERROR: BMD station catalogue not found: $BMD_STATIONS"
    exit 1
}

case "$SLURM_ARRAY_TASK_ID" in
    0) LABEL="gauges_exact_bmd"; MODE="gauges" ;;
    1) LABEL="satellite_exact_bmd"; MODE="satellite" ;;
    2) LABEL="simultaneous_exact_bmd"; MODE="combined" ;;
    *) echo "ERROR: unknown array task $SLURM_ARRAY_TASK_ID"; exit 2 ;;
esac

OUT="$ROOT/$LABEL"
mkdir -p "$OUT"

echo "Final OSSE arm: $LABEL"
echo "mode=$MODE; error=exact; network=BMD; satellite stride=1"
echo "window=$START..$END; months=$MONTHS; days=$DAYS; members=$MEMBERS"
echo "BMD catalogue: $BMD_STATIONS; withhold=$WITHHOLD; seed=$SEED"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/10_osse.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --start "$START" \
    --end "$END" \
    --months "$MONTHS" \
    --days "$DAYS" \
    --members "$MEMBERS" \
    --networks bmd \
    --bmd-stations "$BMD_STATIONS" \
    --station-layout spread \
    --holdout-layout spread \
    --withhold "$WITHHOLD" \
    --seed "$SEED" \
    --obs-error exact \
    --observation-mode "$MODE" \
    --satellite-stride 1 \
    --dump "$OUT/ensemble.npz" \
    --dump-network bmd \
    --dump-obs-error exact \
    --out-figure "$OUT/osse_overview.png" \
    --out-report "$OUT/osse_report.json"

[[ -s "$OUT/ensemble.npz" ]] || {
    echo "ERROR: ensemble dump was not produced: $OUT/ensemble.npz"
    exit 3
}

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

srun --ntasks=1 "$PYTHON_BIN" -u scripts/22_evaluate_osse_downscaling.py \
    --dump "$OUT/ensemble.npz" \
    --label "$LABEL" \
    --minimum-valid-fraction 1.0 \
    --out-figure "$OUT/downscaling_diagnostics.png" \
    --out-report "$OUT/downscaling.json" \
    --out-curve-data "$OUT/downscaling_curves.npz" \
    --out-spatial-data "$OUT/spatial_fields.nc"

if [[ "$MODE" == "combined" ]]; then
    srun --ntasks=1 "$PYTHON_BIN" -u scripts/12_da_impact_maps.py \
        --dump "$OUT/ensemble.npz" \
        --cases 4 \
        --out-cases "$OUT/impact_cases.png" \
        --out-aggregate "$OUT/impact_aggregate.png" \
        --out-report "$OUT/impact.json"
fi

echo "completed $LABEL"
