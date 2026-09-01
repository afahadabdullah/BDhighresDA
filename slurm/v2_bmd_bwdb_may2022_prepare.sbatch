#!/usr/bin/env bash
#SBATCH --job-name=bdhires-bwdb-may-prep
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge 2>/dev/null || true
export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
mkdir -p logs

ROOT="${V2_BWDB_ROOT:-data/processed/v2_bmd_bwdb_may2022}"
START="${V2_BWDB_START:-2022-05-01}"
END="${V2_BWDB_END:-2022-05-31}"
BMD_DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
BMD_STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
BWDB_XLSX="${BWDB_XLSX:-data/stations/BWDB_Rainfall_2000_2025_corrected.xlsx}"
IMERG_ARCHIVE="${V2_BWDB_IMERG_ARCHIVE_GLOB:-data/processed/imerg_bd_aligned_*.nc}"
for required in "$BMD_STATIONS" "$BWDB_XLSX"; do
    [[ -f "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done
[[ -d "$BMD_DATA_DIR" ]] || { echo "ERROR: missing BMD data directory: $BMD_DATA_DIR"; exit 1; }

mkdir -p "$ROOT/stations" "$ROOT/holdouts" "$ROOT/imerg_native" "$ROOT/imerg_s04"
"$PYTHON_BIN" -u scripts/82_prepare_bmd_bwdb_may2022.py \
    --bmd-data-dir "$BMD_DATA_DIR" --bmd-stations "$BMD_STATIONS" --bwdb-xlsx "$BWDB_XLSX" \
    --start "$START" --end "$END" \
    --out "$ROOT/stations/combined_daily.csv" \
    --summary "$ROOT/stations/station_summary.csv" \
    --report "$ROOT/stations/preparation_manifest.json" \
    --holdout-dir "$ROOT/holdouts" \
    --holdout-folds 3 --holdout-fraction 0.20 --holdout-neighbor-km 20 --coincident-group-km 5

# The established prepared IMERG is 03 UTC to 03 UTC.  That is exact for BWDB;
# the manifest records the remaining BMD three-hour support difference.
"$PYTHON_BIN" -u scripts/43_subset_prepared_imerg.py \
    --input "$IMERG_ARCHIVE" --start "$START" --end "$END" \
    --out "$ROOT/imerg_native/may2022.nc" --report "$ROOT/imerg_native/may2022_qc.json"
"$PYTHON_BIN" -u scripts/44_coarsen_imerg_observations.py \
    --input "$ROOT/imerg_native/may2022.nc" --factor 8 \
    --out "$ROOT/imerg_s04/may2022.nc" --report "$ROOT/imerg_s04/may2022_qc.json"

echo "Combined BMD/BWDB May 2022 inputs and constrained 20% folds are ready at $ROOT"
