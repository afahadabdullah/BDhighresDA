#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v2-gauge-sweep
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-4%2
#SBATCH --output=logs/%x-%A_%a.out

# One fold of the CPC-v2 gauges-only DA method tournament.  All methods share
# dates, held-out stations, checkpoint, prior seeds and gauge perturbation
# seeds.  The dependent CPU job pools the five disjoint folds.

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
mkdir -p logs

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: this job must run on an aarch64 Grace/GH200 node" >&2
    exit 1
}

FOLD="${SLURM_ARRAY_TASK_ID:?run as the configured five-fold array}"
START="${V2_SWEEP_START:-2022-05-01}"
END="${V2_SWEEP_END:-2022-05-10}"
MEMBERS="${V2_SWEEP_MEMBERS:-30}"
GROUP="${V2_SWEEP_GROUP:-v2_gauges_core}"
ROOT="${V2_SWEEP_ROOT:-data/processed/v2_gauge_da_sweep/ing2022_core}"
CKPT="${BMD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
CONFIG="${BMD_CONFIG:-configs/da.yaml}"
DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
STATION_CATALOG="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
SEED="${V2_SWEEP_SEED:-201805}"
BACKGROUND_DAY_OFFSET="${BACKGROUND_DAY_OFFSET:--1}"

PREFIX="$ROOT/fold${FOLD}"
DAILY="${PREFIX}_bmd.csv"
STATION_SUMMARY="${PREFIX}_stations.csv"
QC="${PREFIX}_bmd_qc.json"
mkdir -p "$ROOT"

for required in "$CKPT" "$CONFIG" "$STATION_CATALOG"; do
    [[ -f "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done
[[ -d "$DATA_DIR" ]] || { echo "ERROR: BMD data directory missing: $DATA_DIR"; exit 1; }

if [[ ! -s "$DAILY" ]]; then
    "$PYTHON_BIN" -u scripts/05_convert_bmd_dir.py \
        --data-dir "$DATA_DIR" \
        --stations "$STATION_CATALOG" \
        --start "$START" --end "$END" \
        --out "$DAILY" \
        --summary "$STATION_SUMMARY" \
        --report "$QC"
else
    echo "Reusing $DAILY"
fi

SET_ARGS=()
if [[ -n "${BMD_SET:-}" ]]; then
    IFS=';' read -ra SET_PAIRS <<< "$BMD_SET"
    for pair in "${SET_PAIRS[@]}"; do
        SET_ARGS+=(--set "$pair")
    done
fi
STEP_ARGS=()
if [[ -n "${V2_SWEEP_N_STEPS:-}" ]]; then
    STEP_ARGS=(--n-steps "$V2_SWEEP_N_STEPS")
fi

echo "CPC-v2 gauge DA tournament, fold $((FOLD + 1))/5"
echo "  period: $START through $END; members: $MEMBERS"
echo "  group: $GROUP; checkpoint: $CKPT"
echo "  background offset: $BACKGROUND_DAY_OFFSET; output: $PREFIX"
[[ -z "${BMD_SET:-}" ]] || echo "  config overrides: $BMD_SET"

"$PYTHON_BIN" -u scripts/28_simultaneous_method_sweep.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --stations "$DAILY" \
    --start "$START" --end "$END" \
    --background-day-offset "$BACKGROUND_DAY_OFFSET" \
    --members "$MEMBERS" \
    --holdout-folds 5 --holdout-fold "$FOLD" \
    --group "$GROUP" \
    --seed "$SEED" \
    "${SET_ARGS[@]+"${SET_ARGS[@]}"}" \
    "${STEP_ARGS[@]+"${STEP_ARGS[@]}"}" \
    --out "${PREFIX}.npz" \
    --report "${PREFIX}.json"

echo "Fold $FOLD complete: ${PREFIX}.{npz,json}"
