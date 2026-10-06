#!/usr/bin/env bash
#SBATCH --job-name=imerg-hh-production
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/imerg-hh-production-%j.out

set -euo pipefail
module purge 2>/dev/null || true
cd "$SLURM_SUBMIT_DIR"
TASK_ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
TASK_PYTHON="${IMERG_PYTHON:-$TASK_ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
export PYTHONPATH="$PWD/src"
[[ -x "$TASK_PYTHON" ]] || { echo "ERROR: Python missing: $TASK_PYTHON" >&2; exit 1; }
[[ -f "$HOME/.netrc" ]] || { echo "ERROR: Earthdata ~/.netrc missing" >&2; exit 1; }
"$TASK_PYTHON" -c 'import numpy, xarray, netCDF4'
echo "IMERG half-hourly production download; job $SLURM_JOB_ID; host $(hostname)"
if [[ "${1:-}" == "--serial-years-worker" ]]; then
    TASK_SERIAL_START="$2"
    TASK_SERIAL_END="$3"
    TASK_SERIAL_STATE="$4"
    shift 4
    [[ "$TASK_SERIAL_START" =~ ^[0-9]{4}$ && "$TASK_SERIAL_END" =~ ^[0-9]{4}$ ]] || { echo "ERROR: four-digit years required" >&2; exit 2; }
    (( TASK_SERIAL_START >= 2001 && TASK_SERIAL_START <= TASK_SERIAL_END && TASK_SERIAL_END <= 2024 )) || { echo "ERROR: serial years outside 2001..2024" >&2; exit 2; }
    # Keep probe, every reporting year and collection in one Slurm allocation.
    # Year-specific receipts remain compatible with source-check and collection.
    "$TASK_PYTHON" -u scripts/103_download_imerg_production.py "$@" \
        --start-year "$TASK_SERIAL_START" --end-year "$TASK_SERIAL_END" \
        --state "$TASK_SERIAL_STATE" --connections 1 --probe-only
    for ((TASK_SERIAL_YEAR=TASK_SERIAL_START; TASK_SERIAL_YEAR<=TASK_SERIAL_END; TASK_SERIAL_YEAR++)); do
        echo "Serial reporting year $TASK_SERIAL_YEAR; Slurm job $SLURM_JOB_ID"
        "$TASK_PYTHON" -u scripts/103_download_imerg_production.py "$@" \
            --start-year "$TASK_SERIAL_YEAR" --end-year "$TASK_SERIAL_YEAR" \
            --state "$TASK_SERIAL_STATE/years/$TASK_SERIAL_YEAR"
    done
    exec "$TASK_PYTHON" -u scripts/103_download_imerg_production.py "$@" \
        --start-year "$TASK_SERIAL_START" --end-year "$TASK_SERIAL_END" \
        --state "$TASK_SERIAL_STATE" --collect-years
fi
if [[ "${1:-}" == "--year-array-worker" ]]; then
    TASK_ARRAY_START="$2"
    TASK_ARRAY_END="$3"
    TASK_ARRAY_STATE="$4"
    shift 4
    TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Missing year array task ID}"
    [[ "$TASK_INDEX" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid array task ID" >&2; exit 2; }
    TASK_YEAR=$((TASK_ARRAY_START + TASK_INDEX))
    (( TASK_YEAR <= TASK_ARRAY_END )) || { echo "ERROR: array task outside year range" >&2; exit 2; }
    # One transfer at a time per year; the array itself limits total transfers.
    set -- "$@" --start-year "$TASK_YEAR" --end-year "$TASK_YEAR" --connections 1 \
        --state "$TASK_ARRAY_STATE/years/$TASK_YEAR"
    echo "Year-array task $TASK_INDEX: reporting year $TASK_YEAR; one download worker"
fi
exec "$TASK_PYTHON" -u scripts/103_download_imerg_production.py "$@"
