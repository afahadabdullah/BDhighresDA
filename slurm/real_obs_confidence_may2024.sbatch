#!/bin/bash
#SBATCH --job-name=bdhires-real-obs-confidence
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

# The OSSE, rerun with REAL observations.
#
# osse_perfect.npz improved 83-95% of land using a noiseless pseudo-satellite --
# the satellite WAS the truth there. That proved the DA machinery works; it said
# nothing about real IMERG.
#
# This runs the same four arms on 2024-05-01..05 with real IMERG and real BMD,
# at three levels of observation confidence, to find out whether the real system
# underperforms because the observations are bad or because we tell the filter
# they are bad.
#
#   trusted     stride 1, R multiplier 0.25.  Closest analogue of the OSSE's
#               "perfect satellite": use every footprint and believe it.
#   measured    stride 1, R multiplier 1.0.  Every footprint, error as
#               configured. Isolates thinning from weighting.
#   production  stride 3, R multiplier 1.0.  Current configs/da.yaml behaviour,
#               i.e. the settings that produced the multi-year table.
#
# What the arms distinguish:
#   trusted >> production  -> the observations were being thrown away. Retune R
#                             and the stride; this is a config fix, no retrain.
#   trusted ~= production  -> the observation weighting is not the bottleneck and
#                             the wet prior is. Go to the v3 retrain.
#   trusted <  production  -> real IMERG genuinely needs the inflation; the error
#                             model is right and the 6.28x is earned.
#
# Note the fitted IMERG error model measured sigma ~ 1.0 in transformed space
# against configs/da.yaml's guessed 0.35 -- roughly 8x too confident in variance
# -- while the correlation control inflates R by 6.28x in the other direction.
# Two large uncalibrated corrections pointing opposite ways. This separates them.

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

START="${RO_START:-2024-05-01}"
END="${RO_END:-2024-05-05}"
MEMBERS="${RO_MEMBERS:-30}"
FOLDS="${RO_FOLDS:-5}"
FOLD="${RO_FOLD:-0}"
CKPT="${BMD_CKPT:-runs/prior_h100_cpc/best.pt}"
CONFIG="${BMD_CONFIG:-configs/da.yaml}"
OFFSET="${BACKGROUND_DAY_OFFSET:--1}"

ROOT="${RO_ROOT:-data/processed/real_obs_confidence}"
mkdir -p "$ROOT"

STATIONS="$ROOT/bmd_${START//-/}_${END//-/}.csv"
IMERG="$ROOT/imerg_${START//-/}_${END//-/}.nc"

BMD_DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
BMD_STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
IMERG_DIR="${IMERG_DIR:-data/imerg_halfhourly}"

echo "Real-observation confidence test: $START to $END, $MEMBERS members"
echo "Checkpoint $CKPT, background offset $OFFSET, holdout fold $((FOLD+1))/$FOLDS"

# ------------------------------------------------------------- observations
if [[ ! -s "$STATIONS" ]]; then
    "$PYTHON_BIN" -u scripts/05_convert_bmd_dir.py \
        --data-dir "$BMD_DATA_DIR" \
        --stations "$BMD_STATIONS" \
        --start "$START" --end "$END" \
        --out "$STATIONS" \
        --summary "${STATIONS%.csv}_stations.csv" \
        --report "${STATIONS%.csv}_qc.json"
else
    echo "Reusing $STATIONS"
fi

if [[ ! -s "$IMERG" ]]; then
    "$PYTHON_BIN" -u scripts/08_prepare_imerg_observations.py \
        --input "$IMERG_DIR" \
        --source-frequency half-hourly \
        --start "$START" --end "$END" \
        --min-count 48 \
        --accumulation-end-hour-utc 3 \
        --out "$IMERG" \
        --report "${IMERG%.nc}_qc.json" \
        --allow-missing
else
    echo "Reusing $IMERG"
fi

# ------------------------------------------------------------------- arms
# name:stride:r_multiplier
ARMS="${RO_ARMS:-trusted:1:0.25 measured:1:1.0 production:3:1.0}"

for spec in $ARMS; do
    IFS=':' read -r NAME STRIDE RMULT <<< "$spec"
    PREFIX="$ROOT/${NAME}_${START//-/}_${END//-/}"
    echo
    echo "=== arm '$NAME': imerg-stride $STRIDE, imerg-r-multiplier $RMULT ==="

    if [[ -s "${PREFIX}.npz" ]]; then
        echo "Reusing ${PREFIX}.npz"
    else
        "$PYTHON_BIN" -u scripts/15_bmd_month_example.py \
            --config "$CONFIG" \
            --ckpt "$CKPT" \
            --stations "$STATIONS" \
            --imerg "$IMERG" \
            --start "$START" --end "$END" \
            --members "$MEMBERS" \
            --holdout-folds "$FOLDS" --holdout-fold "$FOLD" \
            --background-day-offset "$OFFSET" \
            --imerg-stride "$STRIDE" \
            --imerg-r-multiplier "$RMULT" \
            --out "${PREFIX}.npz" \
            --report "${PREFIX}.json"
    fi

    "$PYTHON_BIN" -u scripts/33_plot_real_da_impact.py \
        --dump "${PREFIX}.npz" \
        --arm combined \
        --days 5 \
        --out-figure "${PREFIX}_impact.png" \
        --out-json "${PREFIX}_impact.json"
done

echo
echo "Done. Compare the withheld-gauge tables across arms:"
for spec in $ARMS; do
    IFS=':' read -r NAME _ _ <<< "$spec"
    echo "  $ROOT/${NAME}_${START//-/}_${END//-/}_impact.json"
done
echo
echo "Read the MEDIAN columns. The mean carries a per-arm Jensen inflation"
echo "(2-6.6 mm/day measured on the multi-year run) and is not comparable."
