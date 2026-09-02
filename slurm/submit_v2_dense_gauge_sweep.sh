#!/usr/bin/env bash
set -euo pipefail

# Dense BMD+BWDB gauge assimilation method sweep, 2022-05-01 .. 2022-05-05.
#
# The frozen v2_simul_s04_huber3 contract was selected on the ~39-station BMD
# network (median nearest-neighbour separation ~42 km) and is now being applied
# to ~303 BMD+BWDB stations (median separation ~13 km, 30 pairs inside two grid
# cells). Three parts of that contract are density-dependent and none were
# re-derived: the 33 km guidance spread kernel, the guidance norm cap, and the
# single scalar gauge error shared by both networks. This sweep re-derives them
# on five days, against the same withheld gauges the production archive uses.
#
# Submit from a Prism login node:
#
#     bash slurm/submit_v2_dense_gauge_sweep.sh
#
# Everything below is overridable from the environment, e.g.
#
#     V2_DENSE_MEMBERS=16 V2_DENSE_END=2022-05-03 \
#         bash slurm/submit_v2_dense_gauge_sweep.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

export V2_DENSE_ROOT="${V2_DENSE_ROOT:-data/processed/v2_dense_gauge_sweep}"
export V2_DENSE_START="${V2_DENSE_START:-2022-05-01}"
export V2_DENSE_END="${V2_DENSE_END:-2022-05-05}"
export V2_DENSE_MEMBERS="${V2_DENSE_MEMBERS:-30}"
export V2_DENSE_SEED="${V2_DENSE_SEED:-202205}"
export V2_DENSE_CELL_DEG="${V2_DENSE_CELL_DEG:-0.25}"
export V2_DENSE_SIGMA_OBS="${V2_DENSE_SIGMA_OBS:-0.10}"
# The 2022 season, seed and constraints of the frozen production archive, so the
# station set and the withheld fold are the production ones rather than a
# five-day re-derivation.
export V2_DENSE_SEASON_START="${V2_DENSE_SEASON_START:-2022-05-01}"
export V2_DENSE_SEASON_END="${V2_DENSE_SEASON_END:-2022-09-30}"
export V2_DENSE_HOLDOUT_SEED="${V2_DENSE_HOLDOUT_SEED:-202122}"
export V2_DENSE_IMERG="${V2_DENSE_IMERG:-data/processed/v2_confirmatory_2021_2024/imerg_s04/2022_may_sep.nc}"
export V2_DENSE_STATS="${V2_DENSE_STATS:-data/processed/stats.json}"

export BMD_CKPT="${BMD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
export BMD_CONFIG="${BMD_CONFIG:-configs/da.yaml}"
export BMD_DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
export BMD_STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
export BWDB_XLSX="${BWDB_XLSX:-data/stations/BWDB_Rainfall_2000_2025_corrected.xlsx}"
export BACKGROUND_DAY_OFFSET="${BACKGROUND_DAY_OFFSET:--1}"

for required in "$BMD_CKPT" "$BMD_CONFIG" "$BMD_STATIONS" "$BWDB_XLSX" \
                "$V2_DENSE_IMERG" "$V2_DENSE_STATS"; do
    [[ -f "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done
[[ -d "$BMD_DATA_DIR" ]] || { echo "ERROR: missing BMD data directory: $BMD_DATA_DIR"; exit 1; }

prepare_result="$(sbatch --parsable --export=ALL "$@" slurm/v2_dense_gauge_sweep_prepare.sbatch)"
prepare_job="${prepare_result%%;*}"
array_result="$(sbatch --parsable --dependency="afterok:${prepare_job}" --export=ALL \
    "$@" slurm/v2_dense_gauge_sweep.sbatch)"
array_job="${array_result%%;*}"
summary_result="$(sbatch --parsable --dependency="afterany:${array_job}" --export=ALL \
    "$@" slurm/v2_dense_gauge_sweep_summary.sbatch)"
summary_job="${summary_result%%;*}"

echo "submitted preparation: $prepare_job (station table, holdout, measured error budget)"
echo "submitted GPU array:   $array_job (6 profiles x one method group each)"
echo "submitted summary:     $summary_job"
echo
echo "period:  $V2_DENSE_START .. $V2_DENSE_END, $V2_DENSE_MEMBERS members"
echo "budget:  $V2_DENSE_ROOT/stations/error_budget.json"
echo "results: $V2_DENSE_ROOT/summary/dense_gauge_sweep.{md,json,csv,png}"
