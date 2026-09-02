#!/usr/bin/env bash
#SBATCH --job-name=bdhires-bwdb-superob
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-7%4
#SBATCH --output=logs/%x-%A_%a.out

# Tasks 0..3: one constrained 20% holdout per seasonal period (superobbed assimilated gauges).
# Tasks 4..7: matched all-station superobbed production ensemble fields.

set -euo pipefail
module purge 2>/dev/null || true
export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
mkdir -p logs
[[ "$(uname -m)" == "aarch64" ]] || { echo "ERROR: Grace/GH200 aarch64 node required"; exit 1; }
"$PYTHON_BIN" -u scripts/51_check_sqrt_da_gradient.py

LABELS=(2021_may_sep 2022_may_sep 2023_may_sep 2024_may_jun)
STARTS=(2021-05-01 2022-05-01 2023-05-01 2024-05-01)
ENDS=(2021-09-30 2022-09-30 2023-09-30 2024-06-30)
TASK_ID="${SLURM_ARRAY_TASK_ID:?configured as 0..7}"
if (( TASK_ID < 4 )); then
    MODE="evaluation"
    PERIOD_INDEX="$TASK_ID"
else
    MODE="production"
    PERIOD_INDEX="$((TASK_ID - 4))"
fi
LABEL="${LABELS[$PERIOD_INDEX]}"
START="${STARTS[$PERIOD_INDEX]}"
END="${ENDS[$PERIOD_INDEX]}"
ROOT="${V2_BWDB_WINNER_ROOT:-data/processed/v2_bmd_bwdb_superob_2021_2024}"
IMERG_ROOT="${V2_BWDB_WINNER_IMERG_ROOT:-data/processed/v2_confirmatory_2021_2024/imerg_s04}"
CELL_DEG="${CELL_DEG:-0.25}"
HOLDOUT="$ROOT/stations/$LABEL/holdout/fold0.txt"
IMERG="$IMERG_ROOT/$LABEL.nc"
CKPT="${BMD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
CONFIG="${BMD_CONFIG:-configs/da.yaml}"
MEMBERS="${V2_BWDB_WINNER_MEMBERS:-30}"
SEED="${V2_BWDB_WINNER_SEED:-202205}"
BACKGROUND_DAY_OFFSET="${BACKGROUND_DAY_OFFSET:--1}"

if [[ "$MODE" == "evaluation" ]]; then
    PREFIX="$ROOT/evaluation/$LABEL"
    STATIONS="$ROOT/stations/$LABEL/superob_eval_${CELL_DEG}.csv"
    SUPEROB_REPORT="$ROOT/stations/$LABEL/superob_eval_${CELL_DEG}.json"
else
    PREFIX="$ROOT/production_metadata/$LABEL"
    FIELD_STORE="$ROOT/gridded/$LABEL.zarr"
    STATIONS="$ROOT/stations/$LABEL/superob_prod_${CELL_DEG}.csv"
    SUPEROB_REPORT="$ROOT/stations/$LABEL/superob_prod_${CELL_DEG}.json"
fi

for required in "$STATIONS" "$SUPEROB_REPORT" "$IMERG" "$CKPT" "$CONFIG"; do
    [[ -f "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done
mkdir -p "$(dirname "$PREFIX")" "$ROOT/gridded"

if [[ "$MODE" == "evaluation" ]]; then
    [[ -f "$HOLDOUT" ]] || { echo "ERROR: holdout missing: $HOLDOUT"; exit 1; }
    if [[ -s "${PREFIX}.npz" && -s "${PREFIX}.json" ]]; then
        echo "$LABEL evaluation already complete; reusing outputs"
        exit 0
    fi
    if [[ -e "${PREFIX}.npz" || -e "${PREFIX}.json" ]]; then
        echo "ERROR: partial evaluation output at $PREFIX" >&2
        exit 1
    fi
else
    if [[ -s "${PREFIX}.npz" && -s "${PREFIX}.json" && -d "$FIELD_STORE" ]]; then
        echo "$LABEL production already complete; reusing outputs"
        exit 0
    fi
    if [[ -e "${PREFIX}.npz" || -e "${PREFIX}.json" || -e "$FIELD_STORE" || -e "${FIELD_STORE}.incomplete" ]]; then
        echo "ERROR: partial production output for $LABEL" >&2
        exit 1
    fi
fi

# Extract measured superob implied representativeness error
SUPEROB_R="$("$PYTHON_BIN" -c "
import json
data = json.load(open('$SUPEROB_REPORT'))
print(data['budget_recommendations']['superob_implied_representativeness'])
")"

COMMON_ARGS=(
    --config "$CONFIG" --ckpt "$CKPT" --stations "$STATIONS" --imerg "$IMERG"
    --start "$START" --end "$END" --background-day-offset "$BACKGROUND_DAY_OFFSET"
    --members "$MEMBERS" --holdout-folds 1 --holdout-fold 0
    --group v2_bmd_bwdb_superob_winner
    --set observations.imerg.factor=8 --set observations.imerg.error_corr_cells=0.75
    --set "observations.gauges.representativeness=$SUPEROB_R"
    --seed "$SEED" --out "${PREFIX}.npz" --report "${PREFIX}.json"
)

echo "$LABEL $MODE: background reference + dense_s6_bwdb_r4; superob_r=$SUPEROB_R; $MEMBERS members"
if [[ "$MODE" == "evaluation" ]]; then
    "$PYTHON_BIN" -u scripts/28_simultaneous_method_sweep.py \
        "${COMMON_ARGS[@]}" --holdout-station-ids-file "$HOLDOUT"
else
    "$PYTHON_BIN" -u scripts/28_simultaneous_method_sweep.py \
        "${COMMON_ARGS[@]}" --assimilate-all-stations --fields-zarr "$FIELD_STORE"
fi
echo "Completed $LABEL $MODE"
