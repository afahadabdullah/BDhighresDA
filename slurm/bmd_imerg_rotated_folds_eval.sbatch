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
export BMD_RAINFALL="${PREFIX}_bmd.csv"
export BMD_STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
export IMERG_DIR="${IMERG_DIR:-data/imerg_halfhourly}"
export IMERG_SOURCE_FREQUENCY="half-hourly"
export BMD_ACCUMULATION_END_HOUR_UTC="3"
export IMERG_MIN_COUNT="48"
export IMERG_STRIDE="${IMERG_STRIDE:-3}"
export IMERG_R_MULTIPLIER="${IMERG_R_MULTIPLIER:-1.0}"
export IMERG_PREPARED="${IMERG_PREPARED:-data/processed/imerg_bd_aligned_20210501_20210930.nc}"
export IMERG_QC="${IMERG_QC:-data/processed/imerg_bd_aligned_20210501_20210930_qc.json}"
export IMERG_REUSE_PREPARED="1"

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
    srun --ntasks=1 "$PYTHON_BIN" -u scripts/05_convert_bmd_dir.py \
        --data-dir "${BMD_DATA_DIR:-data/stations/data_2020_2025}" \
        --stations "${BMD_STATIONS}" \
        --start "$BMD_START" \
        --end "$BMD_END" \
        --out "$BMD_DAILY" \
        --summary "$BMD_STATION_SUMMARY" \
        --report "$BMD_QC_REPORT"
fi

echo "Real BMD + IMERG evaluation, spatial fold $((FOLD + 1))/5"
echo "Period: $BMD_START to $BMD_END ($LABEL)"
echo "Members: $BMD_MEMBERS"
echo "Products: background | gauges only | IMERG only | simultaneous"
exec bash slurm/bmd_imerg_example.sbatch
