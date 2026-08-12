#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v2-confirm
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-23%4
#SBATCH --output=logs/%x-%A_%a.out

# Frozen CPC-v2 confirmation archive:
#   tasks  0..19 = four periods x five withheld-station folds
#   tasks 20..23 = four all-station gridded production analyses

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

# Fail before the seasonal calculation if the exact sqrt-space likelihood
# gradient regresses to the non-finite failure seen in the first IMERG runs.
"$PYTHON_BIN" -u scripts/51_check_sqrt_da_gradient.py

LABELS=(2021_may_sep 2022_may_sep 2023_may_sep 2024_may_jun)
STARTS=(2021-05-01 2022-05-01 2023-05-01 2024-05-01)
ENDS=(2021-09-30 2022-09-30 2023-09-30 2024-06-30)
IMERG_FILES=(
    "${V2_CONFIRM_IMERG_2021:?set V2_CONFIRM_IMERG_2021}"
    "${V2_CONFIRM_IMERG_2022:?set V2_CONFIRM_IMERG_2022}"
    "${V2_CONFIRM_IMERG_2023:?set V2_CONFIRM_IMERG_2023}"
    "${V2_CONFIRM_IMERG_2024:?set V2_CONFIRM_IMERG_2024}"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?run as the configured 24-task array}"
if (( TASK_ID < 20 )); then
    MODE="cv"
    PERIOD_INDEX=$((TASK_ID / 5))
    FOLD=$((TASK_ID % 5))
else
    MODE="production"
    PERIOD_INDEX=$((TASK_ID - 20))
    FOLD="all"
fi
if (( PERIOD_INDEX < 0 || PERIOD_INDEX >= ${#LABELS[@]} )); then
    echo "ERROR: task $TASK_ID maps to invalid period index $PERIOD_INDEX" >&2
    exit 1
fi

LABEL="${LABELS[$PERIOD_INDEX]}"
START="${STARTS[$PERIOD_INDEX]}"
END="${ENDS[$PERIOD_INDEX]}"
IMERG="${IMERG_FILES[$PERIOD_INDEX]}"
ROOT="${V2_CONFIRM_ROOT:-data/processed/v2_confirmatory_2021_2024}"
MEMBERS="${V2_CONFIRM_MEMBERS:-30}"
SEED="${V2_CONFIRM_SEED:-201805}"
CKPT="${BMD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
CONFIG="${BMD_CONFIG:-configs/da.yaml}"
DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
STATION_CATALOG="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
BACKGROUND_DAY_OFFSET="${BACKGROUND_DAY_OFFSET:--1}"

if [[ "$MODE" == "cv" ]]; then
    PREFIX="$ROOT/cv/$LABEL/fold${FOLD}"
else
    PREFIX="$ROOT/production_metadata/$LABEL"
fi
DAILY="${PREFIX}_bmd.csv"
STATION_SUMMARY="${PREFIX}_stations.csv"
QC="${PREFIX}_bmd_qc.json"
mkdir -p "$(dirname "$PREFIX")"

for required in "$CKPT" "$CONFIG" "$STATION_CATALOG" "$IMERG"; do
    [[ -f "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done
[[ -d "$DATA_DIR" ]] || { echo "ERROR: BMD data directory missing: $DATA_DIR"; exit 1; }

if [[ "$MODE" == "cv" ]]; then
    if [[ -s "${PREFIX}.npz" && -s "${PREFIX}.json" ]]; then
        echo "CV $LABEL fold $FOLD already complete; reusing ${PREFIX}.{npz,json}"
        exit 0
    fi
    if [[ -e "${PREFIX}.npz" || -e "${PREFIX}.json" ]]; then
        echo "ERROR: partial CV output at $PREFIX; move it aside before retrying" >&2
        exit 1
    fi
else
    FIELD_STORE="$ROOT/gridded/${LABEL}.zarr"
    mkdir -p "$(dirname "$FIELD_STORE")"
    if [[ -s "${PREFIX}.npz" && -s "${PREFIX}.json" && -d "$FIELD_STORE" ]]; then
        echo "Production $LABEL already complete; reusing $FIELD_STORE"
        exit 0
    fi
    if [[ -e "${PREFIX}.npz" || -e "${PREFIX}.json" || -e "$FIELD_STORE" \
          || -e "${FIELD_STORE}.incomplete" ]]; then
        echo "ERROR: partial production output for $LABEL; inspect or move it aside" >&2
        exit 1
    fi
fi

if [[ ! -s "$DAILY" ]]; then
    "$PYTHON_BIN" -u scripts/05_convert_bmd_dir.py \
        --data-dir "$DATA_DIR" --stations "$STATION_CATALOG" \
        --start "$START" --end "$END" \
        --out "$DAILY" --summary "$STATION_SUMMARY" --report "$QC"
else
    echo "Reusing $DAILY"
fi

echo "CPC-v2 frozen confirmation task $TASK_ID/23"
echo "  mode: $MODE; period: $START through $END; fold: $FOLD"
echo "  methods: background, gauges S6/G0.01, current simultaneous, ig010, huber3"
echo "  members: $MEMBERS; checkpoint: $CKPT"
echo "  IMERG: S04 factor 8, stride 1: $IMERG"
echo "  output prefix: $PREFIX"

COMMON_ARGS=(
    --config "$CONFIG" --ckpt "$CKPT"
    --stations "$DAILY" --imerg "$IMERG"
    --start "$START" --end "$END"
    --background-day-offset "$BACKGROUND_DAY_OFFSET"
    --members "$MEMBERS"
    --holdout-folds 5
    --group v2_confirmatory
    --set observations.imerg.factor=8
    --set observations.imerg.error_corr_cells=0.75
    --seed "$SEED"
    --out "${PREFIX}.npz" --report "${PREFIX}.json"
)

if [[ "$MODE" == "cv" ]]; then
    "$PYTHON_BIN" -u scripts/28_simultaneous_method_sweep.py \
        "${COMMON_ARGS[@]}" --holdout-fold "$FOLD"
    echo "CV complete: ${PREFIX}.{npz,json}"
else
    "$PYTHON_BIN" -u scripts/28_simultaneous_method_sweep.py \
        "${COMMON_ARGS[@]}" --assimilate-all-stations --fields-zarr "$FIELD_STORE"
    echo "Production complete: $FIELD_STORE and ${PREFIX}.{npz,json}"
fi
