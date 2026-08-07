#!/bin/bash
#SBATCH --job-name=bdhires-bmd-imerg
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x-%j.out

# Launcher. On this cluster `srun` inside an allocation fails on the GH200 GPU
# nodes with "CPU binding outside of job step allocation, allocated CPUs are:
# 0x..FF / Unable to satisfy cpu bind request": from SLURM 22.05 the step no
# longer inherits --cpus-per-task from the batch script, so its default binding
# asks for more than the 8 CPUs the job holds. All five folds of the first
# IMERG submission died this way before running a single day.
#
# slurm/cpc_satellite_may2024.sbatch runs Python directly and works, so that is
# the default here too. Set BDHIRES_LAUNCH="run_step --cpu-bind=none" to
# restore a step launcher if a future site needs one.
read -r -a LAUNCH <<< "${BDHIRES_LAUNCH:-}"
run_step() { "${LAUNCH[@]+"${LAUNCH[@]}"}" "$@"; }


# Configurable bounded real-observation experiment: exact 03:00-03:00 UTC BMD
# reporting windows, background, gauges-only, correlation-controlled
# IMERG-only, and stabilized simultaneous DA. BMD stations remain withheld
# from every active assimilation arm.

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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bdhires-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" data/processed logs

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: This job must run on an aarch64 Grace node."
    exit 1
}

RAINFALL="${BMD_RAINFALL:-data/bmd/bmd.csv}"
STATIONS="${BMD_STATIONS:-data/bmd/Stations.csv}"
IMERG_DIR="${IMERG_DIR:-data/imerg_halfhourly}"
IMERG_SOURCE_FREQUENCY="${IMERG_SOURCE_FREQUENCY:-half-hourly}"
BMD_ACCUMULATION_END_HOUR_UTC="${BMD_ACCUMULATION_END_HOUR_UTC:-3}"
START="${BMD_START:-2018-05-01}"
END="${BMD_END:-2018-05-31}"
BACKGROUND_DAY_OFFSET="${BACKGROUND_DAY_OFFSET:-0}"
CKPT="${BMD_CKPT:-runs/prior_h100_cpc/best.pt}"
CONFIG="${BMD_CONFIG:-configs/da.yaml}"
# BMD_SET carries SEMICOLON-separated config overrides through to scripts/15 as
# --set, so a multi-year run can use the tuned configuration without editing the
# YAML. Editing it between submissions is how three arms of the 5-day sweep
# ended up byte-identical to the baseline while appearing to run.
#
# Semicolons, not commas: sbatch --export takes a COMMA-separated list, so a
# comma inside a value silently truncates it into a bogus extra variable.
SET_ARGS=()
if [[ -n "${BMD_SET:-}" ]]; then
    IFS=';' read -ra BMD_SET_PAIRS <<< "$BMD_SET"
    for pair in "${BMD_SET_PAIRS[@]}"; do
        SET_ARGS+=(--set "$pair")
    done
    echo "Config overrides: $BMD_SET"
fi
MEMBERS="${BMD_MEMBERS:-16}"
WITHHOLD="${BMD_WITHHOLD:-0.2}"
HOLDOUT_FOLDS="${BMD_HOLDOUT_FOLDS:-1}"
HOLDOUT_FOLD="${BMD_HOLDOUT_FOLD:-0}"
IMERG_MIN_COUNT="${IMERG_MIN_COUNT:-48}"
IMERG_STRIDE="${IMERG_STRIDE:-3}"
IMERG_R_MULTIPLIER="${IMERG_R_MULTIPLIER:-1.0}"
IMERG_REUSE_PREPARED="${IMERG_REUSE_PREPARED:-0}"

DAILY="${BMD_DAILY:-data/processed/bmd_daily_may2018.csv}"
STATION_SUMMARY="${BMD_STATION_SUMMARY:-data/processed/bmd_stations_may2018.csv}"
BMD_QC="${BMD_QC_REPORT:-data/processed/bmd_qc_may2018.json}"
IMERG_PREPARED="${IMERG_PREPARED:-data/processed/imerg_bd_may2018.nc}"
IMERG_QC="${IMERG_QC:-data/processed/imerg_bd_may2018_qc.json}"
DUMP="${BMD_DUMP:-data/processed/bmd_imerg_may2018_example.npz}"
REPORT="${BMD_REPORT:-data/processed/bmd_imerg_may2018_example.json}"
DIAGNOSTICS="${BMD_DIAGNOSTICS:-data/processed/bmd_imerg_may2018_diagnostics.png}"
EVENTS="${BMD_EVENTS:-data/processed/bmd_imerg_may2018_events.png}"
STATION_COMPARISON="${BMD_STATION_COMPARISON:-data/processed/bmd_imerg_may2018_station_comparison.png}"
SPATIAL="${BMD_SPATIAL:-data/processed/bmd_imerg_may2018_spatial.png}"
INTERCOMPARISON="${BMD_INTERCOMPARISON:-data/processed/bmd_imerg_may2018_intercomparison.png}"
EVALUATION="${BMD_EVALUATION:-data/processed/bmd_imerg_may2018_evaluation.json}"

for required in "$RAINFALL" "$STATIONS" "$CKPT" "$CONFIG"; do
    [[ -f "$required" ]] || { echo "ERROR: required file not found: $required"; exit 1; }
done
[[ -d "$IMERG_DIR" ]] || { echo "ERROR: IMERG directory not found: $IMERG_DIR"; exit 1; }

echo "BMD rainfall: $RAINFALL"
echo "BMD stations: $STATIONS"
echo "IMERG source directory: $IMERG_DIR"
echo "IMERG source frequency: $IMERG_SOURCE_FREQUENCY"
echo "BMD accumulation end: ${BMD_ACCUMULATION_END_HOUR_UTC}:00 UTC"
echo "Period: $START to $END"
echo "Checkpoint background day offset: ${BACKGROUND_DAY_OFFSET}"
echo "Checkpoint: $CKPT"
echo "Members: $MEMBERS   Withhold: $WITHHOLD   IMERG min count: $IMERG_MIN_COUNT"
echo "Holdout fold: $HOLDOUT_FOLD of $HOLDOUT_FOLDS"
echo "Controlled IMERG: stride $IMERG_STRIDE   extra R $IMERG_R_MULTIPLIER"
echo "NOTE: dates through 2018 are in checkpoint training; such runs are process tests, not final skill."
echo "NOTE: native IMERG V07B is not bias-corrected in this bounded experiment."

if [[ "${BMD_REUSE_CONVERTED:-0}" == "1" || ( "$RAINFALL" == "$DAILY" && -s "$DAILY" ) ]]; then
    echo "Reusing prepared canonical station observations: $DAILY"
else
    run_step "$PYTHON_BIN" -u scripts/05_convert_bmd_legacy.py \
        --rainfall "$RAINFALL" \
        --stations "$STATIONS" \
        --start "$START" \
        --end "$END" \
        --out "$DAILY" \
        --summary "$STATION_SUMMARY" \
        --report "$BMD_QC"
fi


if [[ "$IMERG_REUSE_PREPARED" == "1" && -s "$IMERG_PREPARED" && -s "$IMERG_QC" ]]; then
    echo "Reusing validated IMERG preparation: $IMERG_PREPARED"
else
    EXTRA_FLAGS=()
    if [[ "${IMERG_ALLOW_MISSING:-0}" == "1" ]]; then
        EXTRA_FLAGS+=("--allow-missing")
    fi
    run_step "$PYTHON_BIN" -u scripts/08_prepare_imerg_observations.py \
        --input "$IMERG_DIR" \
        --source-frequency "$IMERG_SOURCE_FREQUENCY" \
        --start "$START" \
        --end "$END" \
        --min-count "$IMERG_MIN_COUNT" \
        --accumulation-end-hour-utc "$BMD_ACCUMULATION_END_HOUR_UTC" \
        --out "$IMERG_PREPARED" \
        --report "$IMERG_QC" \
        "${EXTRA_FLAGS[@]}"
fi


run_step "$PYTHON_BIN" -u scripts/15_bmd_month_example.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --stations "$DAILY" \
    --imerg "$IMERG_PREPARED" \
    --start "$START" \
    --end "$END" \
    --background-day-offset "$BACKGROUND_DAY_OFFSET" \
    --members "$MEMBERS" \
    --withhold "$WITHHOLD" \
    --holdout-folds "$HOLDOUT_FOLDS" \
    --holdout-fold "$HOLDOUT_FOLD" \
    --imerg-stride "$IMERG_STRIDE" \
    --imerg-r-multiplier "$IMERG_R_MULTIPLIER" \
    "${SET_ARGS[@]+"${SET_ARGS[@]}"}" \
    --out "$DUMP" \
    --report "$REPORT"

run_step "$PYTHON_BIN" -u scripts/16_plot_bmd_month_example.py \
    --dump "$DUMP" \
    --report "$REPORT" \
    --out-diagnostics "$DIAGNOSTICS" \
    --out-events "$EVENTS" \
    --out-station-comparison "$STATION_COMPARISON" \
    --out-spatial "$SPATIAL" \
    --out-intercomparison "$INTERCOMPARISON" \
    --out-evaluation "$EVALUATION"

echo "BMD + IMERG experiment complete for $START through $END."
echo "Metrics: $REPORT"
echo "IMERG QC: $IMERG_QC"
echo "Diagnostics: $DIAGNOSTICS"
echo "Rainfall-event evaluation: $EVENTS"
echo "Station-collocated product comparison: $STATION_COMPARISON"
echo "Spatial cases: $SPATIAL"
echo "Non-independent product intercomparison: $INTERCOMPARISON"
echo "Withheld-BMD evaluation: $EVALUATION"
