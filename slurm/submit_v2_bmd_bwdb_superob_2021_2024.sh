#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

export V2_BWDB_WINNER_ROOT="${V2_BWDB_WINNER_ROOT:-data/processed/v2_bmd_bwdb_superob_2021_2024}"
export V2_BWDB_WINNER_IMERG_ROOT="${V2_BWDB_WINNER_IMERG_ROOT:-data/processed/v2_confirmatory_2021_2024/imerg_s04}"
export V2_BWDB_WINNER_MEMBERS="${V2_BWDB_WINNER_MEMBERS:-30}"
export V2_BWDB_WINNER_SEED="${V2_BWDB_WINNER_SEED:-202205}"
export V2_BWDB_WINNER_HOLDOUT_SEED="${V2_BWDB_WINNER_HOLDOUT_SEED:-202121}"
export BMD_CKPT="${BMD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
export BMD_CONFIG="${BMD_CONFIG:-configs/da.yaml}"
export BMD_DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
export BMD_STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
export BWDB_XLSX="${BWDB_XLSX:-data/stations/BWDB_Rainfall_2000_2025_corrected.xlsx}"
export CELL_DEG="${CELL_DEG:-0.25}"
export STATS="${STATS:-data/processed/stats.json}"
export BACKGROUND_DAY_OFFSET="${BACKGROUND_DAY_OFFSET:--1}"

for required in "$BMD_CKPT" "$BMD_CONFIG" "$BMD_STATIONS" "$BWDB_XLSX" "$STATS"; do
    [[ -f "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done
[[ -d "$BMD_DATA_DIR" ]] || { echo "ERROR: missing BMD data directory: $BMD_DATA_DIR"; exit 1; }
[[ -d "$V2_BWDB_WINNER_IMERG_ROOT" ]] || { echo "ERROR: missing S04 IMERG root: $V2_BWDB_WINNER_IMERG_ROOT"; exit 1; }

prepare_result="$(sbatch --parsable --export=ALL "$@" slurm/v2_bmd_bwdb_superob_2021_2024_prepare.sbatch)"
prepare_job="${prepare_result%%;*}"
array_result="$(sbatch --parsable --dependency="afterok:${prepare_job}" --export=ALL "$@" slurm/v2_bmd_bwdb_superob_2021_2024.sbatch)"
array_job="${array_result%%;*}"
summary_result="$(sbatch --parsable --dependency="afterok:${array_job}" --export=ALL "$@" slurm/v2_bmd_bwdb_superob_2021_2024_summary.sbatch)"
summary_job="${summary_result%%;*}"

echo "Submitted super-obbed dense production pipeline:"
echo "  Preparation: $prepare_job"
echo "  GPU array:   $array_job (4 constrained evaluations + 4 all-station analyses)"
echo "  Summary:     $summary_job"
echo "Target root:   $V2_BWDB_WINNER_ROOT"
echo "Zarr stores:   $V2_BWDB_WINNER_ROOT/gridded/{2021_may_sep,2022_may_sep,2023_may_sep,2024_may_jun}.zarr"
echo "Summary:       $V2_BWDB_WINNER_ROOT/summary/huber3_2021_2024_scores.{md,json,png}"
