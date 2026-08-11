#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v2-ingest
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-4%2
#SBATCH --output=logs/%x-%A_%a.out

# One fold of the matched CPC-v2 ingestion triplet: selected gauges-only,
# selected IMERG S04-only, and their simultaneous likelihood.

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
START="${V2_INGEST_START:-2022-05-01}"
END="${V2_INGEST_END:-2022-05-10}"
MEMBERS="${V2_INGEST_MEMBERS:-30}"
ROOT="${V2_INGEST_ROOT:-data/processed/v2_ingestion_triplet/ing2022_s04}"
IMERG="${V2_INGEST_IMERG:?set V2_INGEST_IMERG to the prepared S04 file}"
CKPT="${BMD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
CONFIG="${BMD_CONFIG:-configs/da.yaml}"
DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
STATION_CATALOG="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
SEED="${V2_INGEST_SEED:-201805}"
BACKGROUND_DAY_OFFSET="${BACKGROUND_DAY_OFFSET:--1}"

PREFIX="$ROOT/fold${FOLD}"
DAILY="${PREFIX}_bmd.csv"
STATION_SUMMARY="${PREFIX}_stations.csv"
QC="${PREFIX}_bmd_qc.json"
mkdir -p "$ROOT"

for required in "$CKPT" "$CONFIG" "$STATION_CATALOG" "$IMERG"; do
    [[ -f "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done
[[ -d "$DATA_DIR" ]] || { echo "ERROR: BMD data directory missing: $DATA_DIR"; exit 1; }

if [[ ! -s "$DAILY" ]]; then
    "$PYTHON_BIN" -u scripts/05_convert_bmd_dir.py \
        --data-dir "$DATA_DIR" --stations "$STATION_CATALOG" \
        --start "$START" --end "$END" \
        --out "$DAILY" --summary "$STATION_SUMMARY" --report "$QC"
else
    echo "Reusing $DAILY"
fi

echo "CPC-v2 ingestion triplet, fold $((FOLD + 1))/5"
echo "  period: $START through $END; members: $MEMBERS"
echo "  checkpoint: $CKPT"
echo "  IMERG: S04 0.4-degree, factor 8, stride 1: $IMERG"
echo "  background offset: $BACKGROUND_DAY_OFFSET; output: $PREFIX"

"$PYTHON_BIN" -u scripts/28_simultaneous_method_sweep.py \
    --config "$CONFIG" --ckpt "$CKPT" \
    --stations "$DAILY" --imerg "$IMERG" \
    --start "$START" --end "$END" \
    --background-day-offset "$BACKGROUND_DAY_OFFSET" \
    --members "$MEMBERS" \
    --holdout-folds 5 --holdout-fold "$FOLD" \
    --group v2_ingestion_s04 \
    --set observations.imerg.factor=8 \
    --set observations.imerg.error_corr_cells=0.75 \
    --seed "$SEED" \
    --out "${PREFIX}.npz" --report "${PREFIX}.json"

echo "Fold $FOLD complete: ${PREFIX}.{npz,json}"
