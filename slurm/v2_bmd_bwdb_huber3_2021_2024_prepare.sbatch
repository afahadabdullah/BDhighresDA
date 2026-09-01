#!/usr/bin/env bash
#SBATCH --job-name=bdhires-bwdb-h3-prep
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
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

ROOT="${V2_BWDB_WINNER_ROOT:-data/processed/v2_bmd_bwdb_huber3_2021_2024}"
BMD_DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
BMD_STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
BWDB_XLSX="${BWDB_XLSX:-data/stations/BWDB_Rainfall_2000_2025_corrected.xlsx}"
IMERG_ROOT="${V2_BWDB_WINNER_IMERG_ROOT:-data/processed/v2_confirmatory_2021_2024/imerg_s04}"
SEED_BASE="${V2_BWDB_WINNER_HOLDOUT_SEED:-202121}"
LABELS=(2021_may_sep 2022_may_sep 2023_may_sep 2024_may_jun)
STARTS=(2021-05-01 2022-05-01 2023-05-01 2024-05-01)
ENDS=(2021-09-30 2022-09-30 2023-09-30 2024-06-30)

for required in "$BMD_STATIONS" "$BWDB_XLSX"; do
    [[ -f "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done
[[ -d "$BMD_DATA_DIR" ]] || { echo "ERROR: missing BMD data directory: $BMD_DATA_DIR"; exit 1; }

for index in 0 1 2 3; do
    label="${LABELS[$index]}"
    period_root="$ROOT/stations/$label"
    imerg="$IMERG_ROOT/$label.nc"
    [[ -f "$imerg" ]] || { echo "ERROR: frozen S04 IMERG missing: $imerg"; exit 1; }
    mkdir -p "$period_root/holdout"
    "$PYTHON_BIN" -u scripts/82_prepare_bmd_bwdb_may2022.py \
        --bmd-data-dir "$BMD_DATA_DIR" --bmd-stations "$BMD_STATIONS" --bwdb-xlsx "$BWDB_XLSX" \
        --start "${STARTS[$index]}" --end "${ENDS[$index]}" \
        --out "$period_root/combined_daily.csv" \
        --summary "$period_root/station_summary.csv" \
        --report "$period_root/preparation_manifest.json" \
        --holdout-dir "$period_root/holdout" \
        --holdout-folds 1 --holdout-fraction 0.20 \
        --holdout-neighbor-km 15 --coincident-group-km 5 \
        --seed "$((SEED_BASE + index))"
done
echo "Prepared four combined BMD/BWDB periods and one constrained 20% holdout per period."
