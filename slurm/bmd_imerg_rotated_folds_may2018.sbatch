#!/bin/bash
#SBATCH --job-name=bdhires-bmd-may
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

FOLD="${SLURM_ARRAY_TASK_ID}"
PREFIX="data/processed/bmd_imerg_offset_m1_fold${FOLD}_20180501_31"

export BMD_START="2018-05-01"
export BMD_END="2018-05-31"
export BACKGROUND_DAY_OFFSET="-1"
export BMD_HOLDOUT_FOLDS="5"
export BMD_HOLDOUT_FOLD="$FOLD"
export IMERG_DIR="${IMERG_DIR:-data/imerg_halfhourly}"
export IMERG_SOURCE_FREQUENCY="half-hourly"
export BMD_ACCUMULATION_END_HOUR_UTC="3"
export IMERG_MIN_COUNT="${IMERG_MIN_COUNT:-48}"
export IMERG_STRIDE="${IMERG_STRIDE:-3}"
export IMERG_R_MULTIPLIER="${IMERG_R_MULTIPLIER:-1.0}"

export BMD_DAILY="${PREFIX}_bmd.csv"
export BMD_STATION_SUMMARY="${PREFIX}_stations.csv"
export BMD_QC_REPORT="${PREFIX}_bmd_qc.json"
export IMERG_PREPARED="${PREFIX}_imerg.nc"
export IMERG_QC="${PREFIX}_imerg_qc.json"
export BMD_DUMP="${PREFIX}.npz"
export BMD_REPORT="${PREFIX}.json"
export BMD_DIAGNOSTICS="${PREFIX}_diagnostics.png"
export BMD_EVENTS="${PREFIX}_events.png"
export BMD_STATION_COMPARISON="${PREFIX}_station_comparison.png"
export BMD_SPATIAL="${PREFIX}_spatial.png"
export BMD_INTERCOMPARISON="${PREFIX}_intercomparison.png"
export BMD_EVALUATION="${PREFIX}_evaluation.json"

cd "$SLURM_SUBMIT_DIR"
echo "Full-May offset -1 rotated BMD fold $((FOLD + 1))/5"
echo "IMERG observation windows: previous-day 03:00 through selected-day 03:00 UTC"
exec bash slurm/bmd_imerg_example.sbatch
