#!/bin/bash
#SBATCH --job-name=bdhires-method-sweep
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

# Screening sweep over simultaneous-DA methods on a short window.
#
# This is NOT a skill evaluation. Its job is to eliminate methods cheaply so the
# multi-year run is only ever spent on one that has already shown it fixes the
# +10.3 mm/day analysis bias or the station bullseyes. Read the bias, wet-day
# frequency and increment-locality columns; treat small CRPS differences as
# unresolved until the full run says otherwise.

set -euo pipefail

module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: not found: $PYTHON_BIN"; exit 1; }

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bdhires-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" logs

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: This job must run on an aarch64 Grace node."
    exit 1
}

# How to launch each step.
#
# On the GH200 nodes `srun --ntasks=1` inside an already-allocated sbatch can
# fail with "CPU binding outside of job step allocation ... Unable to satisfy
# cpu bind request": srun tries to bind the step to a CPU mask that does not
# match what the job was given. Every step here is a single task on a single
# node, so srun buys nothing -- running the interpreter directly sidesteps the
# step-launch machinery entirely and is the default.
#
# Set SWEEP_USE_SRUN=1 to go back to srun (with binding disabled, which is the
# other working combination) if a site policy requires steps to be launched
# through it.
if [[ "${SWEEP_USE_SRUN:-0}" == "1" ]]; then
    LAUNCH="srun --ntasks=1 --cpu-bind=none --overlap"
    export SLURM_CPU_BIND=none
else
    LAUNCH=""
fi

START="${SWEEP_START:-2024-05-01}"
END="${SWEEP_END:-2024-05-05}"
GROUP="${SWEEP_GROUP:-core}"
MEMBERS="${SWEEP_MEMBERS:-16}"
FOLD="${SWEEP_FOLD:-0}"
FOLDS="${SWEEP_FOLDS:-5}"
CKPT="${BMD_CKPT:-runs/prior_h100_cpc/best.pt}"
CONFIG="${BMD_CONFIG:-configs/da.yaml}"
BACKGROUND_DAY_OFFSET="${BACKGROUND_DAY_OFFSET:--1}"
IMERG_STRIDE="${IMERG_STRIDE:-3}"
IMERG_R_MULTIPLIER="${IMERG_R_MULTIPLIER:-1.0}"

ROOT="${SWEEP_ROOT:-data/processed/method_sweep}"
TAG="${SWEEP_TAG:-$(echo "${START}_${END}" | tr -d '-')_${GROUP}}"
PREFIX="$ROOT/$TAG"
mkdir -p "$ROOT"

STATIONS_DAILY="${SWEEP_STATIONS:-$ROOT/${TAG}_bmd.csv}"
STATION_SUMMARY="$ROOT/${TAG}_stations.csv"
STATION_QC="$ROOT/${TAG}_bmd_qc.json"
IMERG_PREPARED="${SWEEP_IMERG:-$ROOT/${TAG}_imerg.nc}"
IMERG_QC="$ROOT/${TAG}_imerg_qc.json"
IMERG_QM="${SWEEP_IMERG_QM:-data/processed/imerg_qm_loyo.npz}"

BMD_DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
BMD_STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
IMERG_DIR="${IMERG_DIR:-data/imerg_halfhourly}"

for required in "$CKPT" "$CONFIG" "$BMD_STATIONS"; do
    [[ -f "$required" ]] || { echo "ERROR: required file not found: $required"; exit 1; }
done
[[ -d "$IMERG_DIR" ]] || { echo "ERROR: IMERG directory not found: $IMERG_DIR"; exit 1; }

echo "Method sweep: $START to $END, group=$GROUP, members=$MEMBERS"
echo "Checkpoint: $CKPT  (background offset ${BACKGROUND_DAY_OFFSET})"
echo "Holdout fold $((FOLD + 1))/$FOLDS"

# ---------------------------------------------------------------- observations
if [[ ! -s "$STATIONS_DAILY" ]]; then
    $LAUNCH "$PYTHON_BIN" -u scripts/05_convert_bmd_dir.py \
        --data-dir "$BMD_DATA_DIR" \
        --stations "$BMD_STATIONS" \
        --start "$START" --end "$END" \
        --out "$STATIONS_DAILY" \
        --summary "$STATION_SUMMARY" \
        --report "$STATION_QC"
else
    echo "Reusing prepared station observations: $STATIONS_DAILY"
fi

if [[ ! -s "$IMERG_PREPARED" ]]; then
    $LAUNCH "$PYTHON_BIN" -u scripts/08_prepare_imerg_observations.py \
        --input "$IMERG_DIR" \
        --source-frequency half-hourly \
        --start "$START" --end "$END" \
        --min-count 48 \
        --accumulation-end-hour-utc 3 \
        --out "$IMERG_PREPARED" \
        --report "$IMERG_QC" \
        --allow-missing
else
    echo "Reusing prepared IMERG: $IMERG_PREPARED"
fi

# ------------------------------------------------------------- bias correction
# Fitted leave-one-year-out over whatever prepared IMERG already exists, so the
# evaluation year is never present in its own fit. SWEEP_QM_SOURCES should list
# the per-year prepared files; the guard in script 27 refuses to build a map for
# a year it cannot hold out.
if [[ ! -s "$IMERG_QM" ]]; then
    if [[ -n "${SWEEP_QM_SOURCES:-}" ]]; then
        # shellcheck disable=SC2086
        $LAUNCH "$PYTHON_BIN" -u scripts/27_fit_imerg_bias_correction.py \
            --imerg $SWEEP_QM_SOURCES \
            --zarr "${SWEEP_ZARR:-data/processed/bd_wide_cpc.zarr}" \
            --stats "${SWEEP_STATS:-data/processed/stats_cpc.json}" \
            --grid bd --pool "${SWEEP_QM_POOL:-5}" --fit-error-model \
            --out "$IMERG_QM"
    else
        echo "WARNING: no $IMERG_QM and no SWEEP_QM_SOURCES; bias-corrected arms will fail."
    fi
else
    echo "Reusing IMERG quantile map: $IMERG_QM"
fi

# ---------------------------------------------------------------------- sweep
QM_FLAG=()
[[ -s "$IMERG_QM" ]] && QM_FLAG=(--imerg-qm "$IMERG_QM")

$LAUNCH "$PYTHON_BIN" -u scripts/28_simultaneous_method_sweep.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --stations "$STATIONS_DAILY" \
    --imerg "$IMERG_PREPARED" \
    --start "$START" --end "$END" \
    --background-day-offset "$BACKGROUND_DAY_OFFSET" \
    --members "$MEMBERS" \
    --holdout-folds "$FOLDS" --holdout-fold "$FOLD" \
    --imerg-stride "$IMERG_STRIDE" \
    --imerg-r-multiplier "$IMERG_R_MULTIPLIER" \
    --group "$GROUP" \
    "${QM_FLAG[@]}" \
    --out "${PREFIX}.npz" \
    --report "${PREFIX}.json"

$LAUNCH "$PYTHON_BIN" -u scripts/29_summarize_method_sweep.py \
    --dump "${PREFIX}.npz" \
    --report "${PREFIX}.json" \
    --baseline "${SWEEP_BASELINE:-gauges_only}" \
    --out-markdown "${PREFIX}_ranking.md" \
    --out-json "${PREFIX}_ranking.json" \
    --out-plot "${PREFIX}_ranking.png"

echo "Sweep complete."
echo "Ranking table: ${PREFIX}_ranking.md"
echo "Figure:        ${PREFIX}_ranking.png"
