#!/usr/bin/env bash
# Five-day, five-fold real-BMD process test using the CPC-v2 prior.
#
# This is a checkpoint-only substitution relative to the existing real-data
# workflow: configs/da.yaml is unchanged, including guidance.spread_cells=0.
# The run therefore measures what v2 changes without confounding it with a new
# assimilation setting. BMD observations cover 03:00--03:00 UTC; the existing
# timing gate uses the previous checkpoint day, which overlaps 21 of 24 hours.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

START="${BMD_START:-2022-05-01}"
END="${BMD_END:-2022-05-05}"
LABEL="${BMD_EVAL_LABEL:-v2_gauges_20220501_05}"
MEMBERS="${BMD_MEMBERS:-16}"
CKPT="${BMD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
CONFIG="${BMD_CONFIG:-configs/da.yaml}"
DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"

# Preserve optional config overrides without placing them in sbatch's
# comma-delimited --export string. Empty by default for a fair v1/v2 contrast.
export BMD_SET="${BMD_SET:-}"

for required in "$CKPT" "$CONFIG" "$STATIONS"; do
    [[ -f "$required" ]] || { echo "ERROR: required file not found: $required"; exit 1; }
done
[[ -d "$DATA_DIR" ]] || { echo "ERROR: BMD data directory not found: $DATA_DIR"; exit 1; }

ROOT="data/processed/bmd_imerg_eval_${LABEL}"
mkdir -p "$ROOT"

echo "Submitting CPC-v2 real-BMD gauges-only test"
echo "  observations: $START through $END"
echo "  background: previous checkpoint day (offset -1)"
echo "  checkpoint: $CKPT"
echo "  members: $MEMBERS; five disjoint spatial folds"
echo "  output: $ROOT"
if [[ -n "$BMD_SET" ]]; then
    echo "  config overrides: $BMD_SET"
else
    echo "  config overrides: none (checkpoint-only comparison)"
fi

array_result="$(sbatch --parsable \
    --export="ALL,BMD_START=$START,BMD_END=$END,BMD_EVAL_LABEL=$LABEL,BMD_DATA_DIR=$DATA_DIR,BMD_STATIONS=$STATIONS,BMD_MEMBERS=$MEMBERS,BMD_CONFIG=$CONFIG,BMD_CKPT=$CKPT,BMD_GAUGES_ONLY=1" \
    "$@" \
    slurm/bmd_imerg_rotated_folds_eval.sbatch)"
array_job="${array_result%%;*}"

summary_result="$(sbatch --parsable --dependency="afterok:${array_job}" \
    --export="ALL,BMD_EVAL_LABEL=$LABEL" \
    "$@" \
    slurm/bmd_gauges_rotated_folds_summary.sbatch)"
summary_job="${summary_result%%;*}"

echo "submitted five-fold GPU array: $array_job"
echo "submitted dependent gauges-vs-background summary: $summary_job"
echo "SUMMARY_JOB:$summary_job"
