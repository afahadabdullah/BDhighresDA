#!/usr/bin/env bash
#SBATCH --job-name=chirps-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

[[ -f scripts/01_download_chirps.py && -f src/bdhires/grids.py ]] || {
    echo "ERROR: Submit this job from the BDhighresDA repository root."
    exit 1
}

YEAR="${SLURM_ARRAY_TASK_ID:?Submit this file as a Slurm array job.}"
OUT_DIR="${CHIRPS_OUT:-data/raw/chirps}"
PYTHON_BIN="${CHIRPS_PYTHON:-python}"

command -v wget >/dev/null || {
    echo "ERROR: wget is not available on the compute node."
    exit 1
}

command -v "$PYTHON_BIN" >/dev/null || {
    echo "ERROR: Python executable not found: $PYTHON_BIN"
    exit 1
}

"$PYTHON_BIN" - <<'PY'
import xarray

print("xarray:", xarray.__version__)
PY

mkdir -p "$OUT_DIR"

echo "Repository: $SLURM_SUBMIT_DIR"
echo "Year: $YEAR"
echo "Output: $OUT_DIR"
echo "Python: $(command -v "$PYTHON_BIN")"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/01_download_chirps.py \
    --start "$YEAR" \
    --end "$YEAR" \
    --out "$OUT_DIR"
