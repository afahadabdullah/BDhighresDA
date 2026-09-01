#!/usr/bin/env bash
set -euo pipefail

# Submit one CPU preparation job, then the three frozen-CPCv2 constrained
# folds, then a source-stratified score report.  Intended for Prism login.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs
export V2_BWDB_ROOT="${V2_BWDB_ROOT:-data/processed/v2_bmd_bwdb_may2022}"
export V2_BWDB_MEMBERS="${V2_BWDB_MEMBERS:-30}"
export V2_BWDB_SEED="${V2_BWDB_SEED:-202205}"
export BMD_CKPT="${BMD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
export BMD_CONFIG="${BMD_CONFIG:-configs/da.yaml}"
export BMD_DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
export BMD_STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
export BWDB_XLSX="${BWDB_XLSX:-data/stations/BWDB_Rainfall_2000_2025_corrected.xlsx}"
export BACKGROUND_DAY_OFFSET="${BACKGROUND_DAY_OFFSET:--1}"
for required in "$BMD_CKPT" "$BMD_CONFIG" "$BMD_STATIONS" "$BWDB_XLSX"; do
    [[ -f "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done
[[ -d "$BMD_DATA_DIR" ]] || { echo "ERROR: missing BMD directory: $BMD_DATA_DIR"; exit 1; }

prepare_result="$(sbatch --parsable --export=ALL "$@" slurm/v2_bmd_bwdb_may2022_prepare.sbatch)"
prepare_job="${prepare_result%%;*}"
array_result="$(sbatch --parsable --dependency="afterok:${prepare_job}" --export=ALL "$@" slurm/v2_bmd_bwdb_may2022.sbatch)"
array_job="${array_result%%;*}"
summary_result="$(sbatch --parsable --dependency="afterok:${array_job}" --export=ALL "$@" slurm/v2_bmd_bwdb_may2022_summary.sbatch)"
summary_job="${summary_result%%;*}"
echo "submitted preparation: $prepare_job"
echo "submitted GPU folds:  $array_job (folds 0..2, 61 withheld stations each)"
echo "submitted summary:    $summary_job"
echo "results: $V2_BWDB_ROOT/summary/may2022_bmd_bwdb_scores.{md,json}"
