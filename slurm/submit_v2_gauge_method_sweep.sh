#!/usr/bin/env bash
set -euo pipefail

# Submit the five-fold CPC-v2 gauges-only DA tournament and its dependent
# day-block-bootstrap summary.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

export V2_SWEEP_START="${V2_SWEEP_START:-2022-05-01}"
export V2_SWEEP_END="${V2_SWEEP_END:-2022-05-10}"
export V2_SWEEP_MEMBERS="${V2_SWEEP_MEMBERS:-30}"
export V2_SWEEP_GROUP="${V2_SWEEP_GROUP:-v2_gauges_core}"
export V2_SWEEP_LABEL="${V2_SWEEP_LABEL:-ing2022_core}"
export V2_SWEEP_ROOT="${V2_SWEEP_ROOT:-data/processed/v2_gauge_da_sweep/${V2_SWEEP_LABEL}}"
export V2_SWEEP_CURRENT="${V2_SWEEP_CURRENT:-guided_s0_t125}"
# Match scripts/15 and the existing v1/v2 runs so the two legacy-control arms
# reproduce the already inspected outputs rather than introducing seed drift.
export V2_SWEEP_SEED="${V2_SWEEP_SEED:-201805}"

export BMD_CKPT="${BMD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
export BMD_CONFIG="${BMD_CONFIG:-configs/da.yaml}"
export BMD_DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
export BMD_STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
export BACKGROUND_DAY_OFFSET="${BACKGROUND_DAY_OFFSET:--1}"
export BMD_SET="${BMD_SET:-}"

for required in "$BMD_CKPT" "$BMD_CONFIG" "$BMD_STATIONS"; do
    [[ -f "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done
[[ -d "$BMD_DATA_DIR" ]] || { echo "ERROR: missing $BMD_DATA_DIR"; exit 1; }
mkdir -p "$V2_SWEEP_ROOT"

echo "Submitting CPC-v2 gauges-only DA method tournament"
echo "  period: $V2_SWEEP_START through $V2_SWEEP_END"
echo "  group: $V2_SWEEP_GROUP; members: $V2_SWEEP_MEMBERS"
echo "  checkpoint: $BMD_CKPT"
echo "  outputs: $V2_SWEEP_ROOT"
echo "  within-run comparator: $V2_SWEEP_CURRENT"
echo "  final comparison: $V2_SWEEP_ROOT/method_selection.{md,json,png}"

array_result="$(sbatch --parsable --export=ALL "$@" slurm/v2_gauge_method_sweep.sbatch)"
array_job="${array_result%%;*}"
summary_result="$(sbatch --parsable --dependency="afterok:${array_job}" --export=ALL \
    "$@" slurm/v2_gauge_method_sweep_summary.sbatch)"
summary_job="${summary_result%%;*}"

echo "submitted five-fold GPU array: $array_job"
echo "submitted dependent pooled summary: $summary_job"
echo "SUMMARY_JOB:$summary_job"
