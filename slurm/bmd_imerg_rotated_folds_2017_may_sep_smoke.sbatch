#!/usr/bin/env bash
#SBATCH --job-name=bdhires-bmd-2017-smoke
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
ROOT="${BMD_2017_SMOKE_ROOT:-data/processed/bmd_imerg_smoke_2017_may_sep}"
PREFIX="$ROOT/fold${FOLD}"

export BMD_START="2017-05-01"
export BMD_END="2017-09-30"
export BACKGROUND_DAY_OFFSET="-1"
export BMD_HOLDOUT_FOLDS="5"
export BMD_HOLDOUT_FOLD="$FOLD"
export BMD_MEMBERS="${BMD_MEMBERS:-4}"
export BMD_RAINFALL="${BMD_RAINFALL:-data/bmd/bmd.csv}"
export BMD_STATIONS="${BMD_STATIONS:-data/bmd/Stations.csv}"
export IMERG_DIR="${IMERG_DIR:-data/imerg_halfhourly/2017}"
export IMERG_SOURCE_FREQUENCY="half-hourly"
export BMD_ACCUMULATION_END_HOUR_UTC="3"
export IMERG_MIN_COUNT="48"
export IMERG_STRIDE="${IMERG_STRIDE:-3}"
export IMERG_R_MULTIPLIER="${IMERG_R_MULTIPLIER:-1.0}"
export IMERG_PREPARED="$ROOT/imerg_aligned_20170501_20170930.nc"
export IMERG_QC="$ROOT/imerg_aligned_20170501_20170930_qc.json"
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
[[ -s "$IMERG_PREPARED" ]] || { echo "ERROR: missing $IMERG_PREPARED"; exit 1; }
[[ -s "$IMERG_QC" ]] || { echo "ERROR: missing $IMERG_QC"; exit 1; }

echo "2017 May-September process test, spatial fold $((FOLD + 1))/5"
echo "Members: $BMD_MEMBERS (low-member smoke configuration; not a final skill run)"
echo "Products: background | gauges only | IMERG only | simultaneous"
echo "IMERG/BMD windows: previous-day 03:00 through selected-day 03:00 UTC"
exec bash slurm/bmd_imerg_example.sbatch
