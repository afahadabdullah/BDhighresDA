#!/usr/bin/env bash
#SBATCH --job-name=bdhires-bmd-eval
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-4%2
#SBATCH --output=logs/%x-%A_%a.out

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


set -euo pipefail

FOLD="${SLURM_ARRAY_TASK_ID:?This job must run as the configured five-fold array}"
LABEL="${BMD_EVAL_LABEL:-2021_may_sep}"
ROOT="data/processed/bmd_imerg_eval_${LABEL}"
PREFIX="$ROOT/fold${FOLD}"

export BMD_START="${BMD_START:-2021-05-01}"
export BMD_END="${BMD_END:-2021-09-30}"
export BACKGROUND_DAY_OFFSET="-1"
export BMD_HOLDOUT_FOLDS="5"
export BMD_HOLDOUT_FOLD="$FOLD"
export BMD_MEMBERS="${BMD_MEMBERS:-16}"
export BMD_GAUGES_ONLY="${BMD_GAUGES_ONLY:-0}"
export BMD_RAINFALL="${PREFIX}_bmd.csv"
export BMD_STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
START_CLEAN="${BMD_START//-/}"
END_CLEAN="${BMD_END//-/}"

export IMERG_DIR="${IMERG_DIR:-data/imerg_halfhourly}"
export IMERG_SOURCE_FREQUENCY="half-hourly"
export BMD_ACCUMULATION_END_HOUR_UTC="3"
export IMERG_MIN_COUNT="48"
export IMERG_STRIDE="${IMERG_STRIDE:-3}"
export IMERG_R_MULTIPLIER="${IMERG_R_MULTIPLIER:-1.0}"
export IMERG_PREPARED="${IMERG_PREPARED:-$ROOT/imerg_aligned_${START_CLEAN}_${END_CLEAN}.nc}"
export IMERG_QC="${IMERG_QC:-$ROOT/imerg_aligned_${START_CLEAN}_${END_CLEAN}_qc.json}"

export IMERG_REUSE_PREPARED="1"
export IMERG_ALLOW_MISSING="1"
export BMD_REUSE_CONVERTED="1"



export BMD_DAILY="${PREFIX}_bmd.csv"
export BMD_STATION_SUMMARY="${PREFIX}_stations.csv"
export BMD_QC_REPORT="${PREFIX}_bmd_qc.json"
export BMD_DUMP="${PREFIX}.npz"
export BMD_REPORT="${PREFIX}.json"
export BMD_DIAGNOSTICS="${PREFIX}_diagnostics.png"
export BMD_EVENTS="${PREFIX}_events.png"
export BMD_STATION_COMPARISON="${PREFIX}_station_comparison.png"
export BMD_SPATIAL="${PREFIX}_spatial.png"
export BMD_INTERCOMPARISON="${PREFIX}_intercomparison.png"
export BMD_EVALUATION="${PREFIX}_evaluation.json"

cd "$SLURM_SUBMIT_DIR"
mkdir -p "$ROOT" logs

# Ensure station data is converted for this period if not done yet
ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
if [[ ! -s "$BMD_DAILY" ]]; then
    run_step "$PYTHON_BIN" -u scripts/05_convert_bmd_dir.py \
        --data-dir "${BMD_DATA_DIR:-data/stations/data_2020_2025}" \
        --stations "${BMD_STATIONS}" \
        --start "$BMD_START" \
        --end "$BMD_END" \
        --out "$BMD_DAILY" \
        --summary "$BMD_STATION_SUMMARY" \
        --report "$BMD_QC_REPORT"
fi

if [[ "$BMD_GAUGES_ONLY" == "1" ]]; then
    echo "Real BMD gauges-only evaluation, spatial fold $((FOLD + 1))/5"
else
    echo "Real BMD + IMERG evaluation, spatial fold $((FOLD + 1))/5"
fi
echo "Period: $BMD_START to $BMD_END ($LABEL)"
echo "Members: $BMD_MEMBERS"
if [[ "$BMD_GAUGES_ONLY" == "1" ]]; then
    echo "Products: checkpoint background | BMD-gauge analysis"
else
    echo "Products: background | gauges only | IMERG only | simultaneous"
fi
exec bash slurm/bmd_imerg_example.sbatch
