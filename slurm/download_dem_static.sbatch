#!/usr/bin/env bash
#SBATCH --job-name=dem-static
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

[[ -f scripts/03_download_dem.py && -f scripts/03_build_static.py ]] || {
    echo "ERROR: Submit this job from the BDhighresDA repository root."
    exit 1
}

DEM_OUT="${DEM_OUT:-data/raw/dem/copernicus_glo90_wide.nc}"
DEM_TILE_DIR="${DEM_TILE_DIR:-data/raw/dem/copernicus_glo90_tiles}"
STATIC_OUT="${STATIC_OUT:-data/static/static_wide.nc}"
CHIRPS_REFERENCE="${CHIRPS_REFERENCE:-data/raw/chirps/chirps_wide_2010.nc}"
JOBS="${DEM_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"

[[ -f "$CHIRPS_REFERENCE" ]] || {
    echo "ERROR: CHIRPS reference file not found: $CHIRPS_REFERENCE"
    echo "Set CHIRPS_REFERENCE to any completed chirps_wide_YEAR.nc file."
    exit 1
}

export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

if [[ -n "${DEM_PYTHON:-}" ]]; then
    PYTHON_BIN="$DEM_PYTHON"
else
    PYTHON_BIN=""
    PYTHON_CANDIDATES=(
        "$SLURM_SUBMIT_DIR/../envs/bdda-gh200/bin/python"
        "/home/afahad/nb/project/BDDA/envs/bdda-gh200/bin/python"
    )
    for candidate in "${PYTHON_CANDIDATES[@]}"; do
        if [[ -x "$candidate" ]]; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
    [[ -n "$PYTHON_BIN" ]] || {
        echo "ERROR: ARM-native project Python was not found."
        echo "Set DEM_PYTHON=/absolute/path/to/ARM/python and resubmit."
        exit 1
    }
fi

"$PYTHON_BIN" - <<'PY'
import netCDF4
import numpy
import rasterio
import xarray

print("numpy:", numpy.__version__)
print("xarray:", xarray.__version__)
print("rasterio:", rasterio.__version__)
print("netCDF4:", netCDF4.__version__)
PY

echo "Repository: $SLURM_SUBMIT_DIR"
echo "Architecture: $(uname -m)"
echo "DEM: $DEM_OUT"
echo "Static fields: $STATIC_OUT"
echo "CHIRPS mask source: $CHIRPS_REFERENCE"
echo "Python: $PYTHON_BIN"
echo "Workers: $JOBS"

DEM_ARGS=(
    --out "$DEM_OUT"
    --tile-dir "$DEM_TILE_DIR"
    --jobs "$JOBS"
)
if [[ "${DEM_KEEP_TILES:-0}" == "1" ]]; then
    DEM_ARGS+=(--keep-tiles)
fi

srun --ntasks=1 "$PYTHON_BIN" -u scripts/03_download_dem.py "${DEM_ARGS[@]}"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/03_build_static.py \
    --dem "$DEM_OUT" \
    --chirps "$CHIRPS_REFERENCE" \
    --out "$STATIC_OUT"
