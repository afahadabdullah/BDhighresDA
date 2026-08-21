#!/usr/bin/env bash
set -euo pipefail

# Submit the one-fold May 1-10 CPC-v2 gauge-authority screen.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

export V2_GW_START="${V2_GW_START:-2022-05-01}"
export V2_GW_END="${V2_GW_END:-2022-05-10}"
export V2_GW_MEMBERS="${V2_GW_MEMBERS:-30}"
export V2_GW_FOLD="${V2_GW_FOLD:-0}"
export V2_GW_FOLDS="${V2_GW_FOLDS:-5}"
export V2_GW_SEED="${V2_GW_SEED:-201805}"
export V2_GW_ROOT="${V2_GW_ROOT:-data/processed/v2_gauge_authority_may2022}"
export V2_GW_IMERG_SOURCE="${V2_GW_IMERG_SOURCE:-data/processed/v2_confirmatory_2021_2024/imerg_s04/2022_may_sep.nc}"
export BMD_CKPT="${BMD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
export BMD_CONFIG="${BMD_CONFIG:-configs/da.yaml}"
export BMD_DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
export BMD_STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
export BACKGROUND_DAY_OFFSET="${BACKGROUND_DAY_OFFSET:--1}"

for required in "$BMD_CKPT" "$BMD_CONFIG" "$BMD_STATIONS" "$V2_GW_IMERG_SOURCE"; do
    [[ -f "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done
[[ -d "$BMD_DATA_DIR" ]] || { echo "ERROR: missing $BMD_DATA_DIR"; exit 1; }

echo "Submitting one-fold CPC-v2 gauge-authority screen"
echo "  period: $V2_GW_START through $V2_GW_END"
echo "  fold: $((V2_GW_FOLD + 1))/$V2_GW_FOLDS; members: $V2_GW_MEMBERS"
echo "  checkpoint: $BMD_CKPT"
echo "  production S04 IMERG: $V2_GW_IMERG_SOURCE"
echo "  outputs: $V2_GW_ROOT/fold${V2_GW_FOLD}_matrix.{md,json,png}"

submit_result="$(sbatch --parsable --export=ALL "$@" slurm/v2_gauge_authority_may2022.sbatch)"
job_id="${submit_result%%;*}"

echo "submitted GPU job: $job_id"
echo "monitor: squeue -j $job_id"
echo "log:     logs/bdhires-v2-gauge-weight-${job_id}.out"
