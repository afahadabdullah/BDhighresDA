#!/usr/bin/env bash
#SBATCH --job-name=pack-training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

[[ -f scripts/04_regrid_and_pack.py && -f scripts/04_check_alignment.py ]] || {
    echo "ERROR: Submit this job from the BDhighresDA repository root."
    exit 1
}

START="${PACK_START:-1981}"
END="${PACK_END:-2025}"
ERA5_DIR="${PACK_ERA5:-data/raw/era5}"
CHIRPS_DIR="${PACK_CHIRPS:-data/raw/chirps}"
STATIC_FILE="${PACK_STATIC:-data/static/static_wide.nc}"
OUTPUT="${PACK_OUT:-data/processed/bd_wide.zarr}"
QC_OUTPUT="${PACK_QC_OUT:-data/processed/alignment_qc.json}"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
unset PYTHONHOME
unset PYTHONPATH

if [[ -n "${PACK_PYTHON:-}" ]]; then
    PYTHON_BIN="$PACK_PYTHON"
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
        echo "Set PACK_PYTHON=/absolute/path/to/ARM/python and resubmit."
        exit 1
    }
fi

"$PYTHON_BIN" - <<'PY'
import netCDF4
import numcodecs
import numpy
import pandas
import xarray
import zarr

print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("xarray:", xarray.__version__)
print("zarr:", zarr.__version__)
print("numcodecs:", numcodecs.__version__)
print("netCDF4:", netCDF4.__version__)
PY

mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$QC_OUTPUT")"

echo "Repository: $SLURM_SUBMIT_DIR"
echo "Architecture: $(uname -m)"
echo "Years: $START-$END"
echo "ERA5: $ERA5_DIR"
echo "CHIRPS: $CHIRPS_DIR"
echo "Static fields: $STATIC_FILE"
echo "Packed store: $OUTPUT"
echo "Alignment report: $QC_OUTPUT"
echo "Python: $PYTHON_BIN"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/04_regrid_and_pack.py \
    --start "$START" \
    --end "$END" \
    --era5 "$ERA5_DIR" \
    --chirps "$CHIRPS_DIR" \
    --static "$STATIC_FILE" \
    --out "$OUTPUT"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/04_check_alignment.py \
    --zarr "$OUTPUT" \
    --out "$QC_OUTPUT"
