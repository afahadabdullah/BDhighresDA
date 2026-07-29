#!/usr/bin/env bash
#SBATCH --job-name=era5-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

[[ -f scripts/00_download_era5.py && -f src/bdhires/grids.py ]] || {
    echo "ERROR: Submit this job from the BDhighresDA repository root."
    exit 1
}

START_YEAR="${ERA5_START_YEAR:?Use slurm/submit_download_era5.sh to submit this job.}"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?Submit this file as a Slurm array job.}"
YEAR="$((START_YEAR + ARRAY_INDEX))"
OUT_DIR="${ERA5_OUT:-data/raw/era5}"
WORKERS="${ERA5_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"

export PYTHONNOUSERSITE=1
export AWS_EC2_METADATA_DISABLED=true
unset PYTHONHOME
unset PYTHONPATH

if [[ -n "${ERA5_PYTHON:-}" ]]; then
    PYTHON_BIN="$ERA5_PYTHON"
else
    PYTHON_BIN=""
    PYTHON_CANDIDATES=(
        "$SLURM_SUBMIT_DIR/../envs/bdda-earthmover/bin/python"
        "/home/afahad/nb/project/BDDA/envs/bdda-earthmover/bin/python"
    )
    for candidate in "${PYTHON_CANDIDATES[@]}"; do
        if [[ -x "$candidate" ]]; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
    [[ -n "$PYTHON_BIN" ]] || {
        echo "ERROR: The dedicated Python 3.12 Earthmover environment was not found."
        echo "From the repository root, create it on an ARM CPU node with:"
        echo "  slurm/setup_earthmover_env.sh"
        echo "or set ERA5_PYTHON=/absolute/path/to/python3.12."
        exit 1
    }
fi

command -v "$PYTHON_BIN" >/dev/null || {
    echo "ERROR: Python executable not found: $PYTHON_BIN"
    exit 1
}

"$PYTHON_BIN" - <<'PY'
import sys
from importlib.metadata import version

assert sys.version_info >= (3, 12), sys.version
import dask
import icechunk
import netCDF4
import pcodec
import xarray
import zarr

print("python:", sys.version.split()[0])
print("icechunk:", version("icechunk"))
print("pcodec:", version("pcodec"))
print("xarray:", xarray.__version__)
print("zarr:", zarr.__version__)
print("dask:", dask.__version__)
print("netCDF4:", netCDF4.__version__)
PY

mkdir -p "$OUT_DIR"

echo "Repository: $SLURM_SUBMIT_DIR"
echo "Architecture: $(uname -m)"
echo "Array index: $ARRAY_INDEX"
echo "Year: $YEAR"
echo "Output: $OUT_DIR"
echo "Python: $(command -v "$PYTHON_BIN")"
echo "Workers: $WORKERS"

ERA5_ARGS=(
    --start "$YEAR"
    --end "$YEAR"
    --out "$OUT_DIR"
    --workers "$WORKERS"
)

srun --ntasks=1 "$PYTHON_BIN" -u scripts/00_download_era5.py "${ERA5_ARGS[@]}"
