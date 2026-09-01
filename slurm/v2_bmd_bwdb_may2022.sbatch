#!/usr/bin/env bash
#SBATCH --job-name=bdhires-bwdb-may
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/%x-%A_%a.out

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

ROOT="${V2_BWDB_ROOT:-data/processed/v2_bmd_bwdb_may2022}"
FOLD="${SLURM_ARRAY_TASK_ID:?configured as fold 0..2}"
START="${V2_BWDB_START:-2022-05-01}"
END="${V2_BWDB_END:-2022-05-31}"
CKPT="${BMD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
CONFIG="${BMD_CONFIG:-configs/da.yaml}"
MEMBERS="${V2_BWDB_MEMBERS:-30}"
SEED="${V2_BWDB_SEED:-202205}"
BACKGROUND_DAY_OFFSET="${BACKGROUND_DAY_OFFSET:--1}"
STATIONS="$ROOT/stations/combined_daily.csv"
HOLDOUT="$ROOT/holdouts/fold${FOLD}.txt"
IMERG="$ROOT/imerg_s04/may2022.nc"
PREFIX="$ROOT/cv/fold${FOLD}"
for required in "$CKPT" "$CONFIG" "$STATIONS" "$HOLDOUT" "$IMERG"; do
    [[ -f "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done
mkdir -p "$ROOT/cv"
if [[ -s "${PREFIX}.npz" && -s "${PREFIX}.json" ]]; then
    echo "Fold $FOLD is already complete; reusing ${PREFIX}.{npz,json}"
    exit 0
fi
if [[ -e "${PREFIX}.npz" || -e "${PREFIX}.json" ]]; then
    echo "ERROR: partial output at $PREFIX; inspect before retrying" >&2
    exit 1
fi

"$PYTHON_BIN" -u scripts/28_simultaneous_method_sweep.py \
    --config "$CONFIG" --ckpt "$CKPT" --stations "$STATIONS" --imerg "$IMERG" \
    --start "$START" --end "$END" --background-day-offset "$BACKGROUND_DAY_OFFSET" \
    --members "$MEMBERS" --holdout-folds 3 --holdout-fold "$FOLD" \
    --holdout-station-ids-file "$HOLDOUT" --group v2_confirmatory \
    --set observations.imerg.factor=8 --set observations.imerg.error_corr_cells=0.75 \
    --seed "$SEED" --out "${PREFIX}.npz" --report "${PREFIX}.json"
echo "Completed constrained BMD/BWDB May 2022 fold $FOLD"
