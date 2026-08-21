#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v7-jun23
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-5%3
#SBATCH --output=logs/%x-%A_%a.out

# Frozen June 2023 V7 R81 pilot:
#   tasks 0..4 = the exact CPCv2 2023 withheld-station folds
#   task  5    = all-station physical ensemble Zarr

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: ARM Python not found: $PYTHON_BIN" >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD/src"
mkdir -p logs

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: this job must run on an aarch64 Grace/GH200 node" >&2
    exit 1
}

TASK_ID="${SLURM_ARRAY_TASK_ID:?run as the configured six-task array}"
ROOT="${V7_JUNE_ROOT:-data/processed/v7_june2023_r81_latest_latest}"
MODEL_START="2023-05-31"
MODEL_END="2023-06-29"
OBS_START="2023-06-01"
OBS_END="2023-06-30"
MEMBERS="${V7_JUNE_MEMBERS:-30}"
STEPS="${V7_JUNE_STEPS:-50}"
SEED="${V7_JUNE_SEED:-201805}"

# These are the immutable files captured by the May-3 tournament, not the live
# best.pt symlinks/checkpoints that training may replace later.
TOURNAMENT="${V7_JUNE_TOURNAMENT:-data/processed/v7_checkpoint_tournament_may03/20260821_1323}"
MESO_CKPT="${V7_JUNE_MESO_CKPT:-$TOURNAMENT/source_checkpoints/latest_meso.pt}"
ALLOC_CKPT="${V7_JUNE_ALLOC_CKPT:-$TOURNAMENT/source_checkpoints/latest_allocation.pt}"
MESO_ARCHIVE="${V7_MESO_ARCHIVE:-data/processed/bd_wide_cpc_0p1.zarr}"
MESO_STATS="${V7_MESO_STATS:-data/processed/stats_v7_meso.json}"
SUBGRID="${V7_PREP_OUT:-data/processed/v7/wide_v7.zarr}"
DAILY="$ROOT/inputs/bmd_daily_202306.csv"
IMERG="$ROOT/inputs/imerg_native_20230601_20230630.nc"
STATION_POOL="$ROOT/comparison_contract/cpcv2_2023_station_ids.txt"

for required in "$MESO_CKPT" "$ALLOC_CKPT" "$MESO_ARCHIVE" "$MESO_STATS" \
                "$SUBGRID" "$DAILY" "$IMERG" "$STATION_POOL" \
                scripts/51_check_sqrt_da_gradient.py scripts/72_v7_two_stage_osse.py; do
    [[ -f "$required" || -d "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done

"$PYTHON_BIN" -u scripts/51_check_sqrt_da_gradient.py

if (( TASK_ID < 5 )); then
    MODE="cv"
    FOLD="$TASK_ID"
    DESTINATION="$ROOT/cv/june2023/fold${FOLD}"
    HOLDOUT="$ROOT/comparison_contract/cpcv2_2023_fold${FOLD}_withheld_ids.txt"
    [[ -s "$HOLDOUT" ]] || { echo "ERROR: missing $HOLDOUT" >&2; exit 1; }
else
    MODE="production"
    FOLD="all"
    DESTINATION="$ROOT/production/june2023"
fi
STATION_DUMP="$DESTINATION/station_ensembles.npz"
REPORT="$DESTINATION/v7_two_stage_real.json"

if [[ "$MODE" == "production" ]]; then
    FIELD_STORE="$ROOT/gridded/june2023.zarr"
    if [[ -s "$STATION_DUMP" && -s "$REPORT" && -d "$FIELD_STORE" ]]; then
        echo "V7 June production already complete; reusing $FIELD_STORE"
        exit 0
    fi
    if [[ -e "$STATION_DUMP" || -e "$REPORT" || -e "$FIELD_STORE" \
          || -e "${FIELD_STORE}.incomplete" ]]; then
        echo "ERROR: partial production output exists; inspect $DESTINATION and $FIELD_STORE" >&2
        exit 1
    fi
else
    if [[ -s "$STATION_DUMP" && -s "$REPORT" ]]; then
        echo "V7 June fold $FOLD already complete; reusing $DESTINATION"
        exit 0
    fi
    if [[ -e "$STATION_DUMP" || -e "$REPORT" ]]; then
        echo "ERROR: partial fold output exists at $DESTINATION" >&2
        exit 1
    fi
fi

mkdir -p "$DESTINATION"
echo "V7 June 2023 R81 task $TASK_ID/5"
echo "  mode: $MODE; fold: $FOLD"
echo "  model/CHIRPS dates: $MODEL_START through $MODEL_END"
echo "  BMD/IMERG/archive dates: $OBS_START through $OBS_END"
echo "  members: $MEMBERS; steps: $STEPS; seed: $SEED"
echo "  meso checkpoint: $MESO_CKPT"
echo "  allocation checkpoint: $ALLOC_CKPT"
echo "  only arm: da_sim_r81"

COMMON_ARGS=(
    --meso-checkpoint "$MESO_CKPT"
    --allocation-checkpoint "$ALLOC_CKPT"
    --meso-archive "$MESO_ARCHIVE"
    --meso-stats "$MESO_STATS"
    --subgrid-archive "$SUBGRID"
    --stations "$DAILY"
    --station-ids-file "$STATION_POOL"
    --imerg "$IMERG"
    --start "$MODEL_START" --end "$MODEL_END"
    --members "$MEMBERS" --n-steps "$STEPS" --seed "$SEED"
    --observations real --min-coverage 0.8
    --arms background --imerg-r-only 81
    --station-dump "$STATION_DUMP"
    --out "$DESTINATION" --no-plots
)

if [[ "$MODE" == "cv" ]]; then
    srun --ntasks=1 "$PYTHON_BIN" -u scripts/72_v7_two_stage_osse.py \
        "${COMMON_ARGS[@]}" --holdout-station-ids-file "$HOLDOUT"
else
    srun --ntasks=1 "$PYTHON_BIN" -u scripts/72_v7_two_stage_osse.py \
        "${COMMON_ARGS[@]}" --assimilate-all-stations --fields-zarr "$FIELD_STORE"
fi

echo "complete: $DESTINATION"
[[ "$MODE" == "production" ]] && echo "physical ensemble: $FIELD_STORE"
