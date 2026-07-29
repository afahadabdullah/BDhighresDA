#!/usr/bin/env bash
#SBATCH --job-name=era5-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

[[ -f scripts/00_download_era5.py && -f src/bdhires/grids.py ]] || {
    echo "ERROR: Submit this job from the BDhighresDA repository root."
    exit 1
}

[[ -s "$HOME/.cdsapirc" ]] || {
    echo "ERROR: Missing $HOME/.cdsapirc."
    echo "Create it from https://cds.climate.copernicus.eu/profile and resubmit."
    exit 1
}

START_YEAR="${ERA5_START_YEAR:?Use slurm/submit_download_era5.sh to submit this job.}"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?Submit this file as a Slurm array job.}"
YEAR="$((START_YEAR + ARRAY_INDEX))"
OUT_DIR="${ERA5_OUT:-data/raw/era5}"

export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

if [[ -n "${ERA5_PYTHON:-}" ]]; then
    PYTHON_BIN="$ERA5_PYTHON"
elif [[ "$(uname -m)" == "aarch64" ]]; then
    PYTHON_BIN=""
    ARM_PYTHON_CANDIDATES=(
        "$SLURM_SUBMIT_DIR/../envs/bdda-gh200/bin/python"
        "/home/afahad/nb/project/BDDA/envs/bdda-gh200/bin/python"
    )
    for candidate in "${ARM_PYTHON_CANDIDATES[@]}"; do
        if [[ -x "$candidate" ]]; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
    [[ -n "$PYTHON_BIN" ]] || {
        echo "ERROR: ARM-native project Python was not found."
        echo "Set ERA5_PYTHON=/absolute/path/to/ARM/python and resubmit."
        exit 1
    }
else
    PYTHON_BIN="$(command -v python || command -v python3 || true)"
fi

command -v "$PYTHON_BIN" >/dev/null || {
    echo "ERROR: Python executable not found: $PYTHON_BIN"
    exit 1
}

"$PYTHON_BIN" - <<'PY'
import cdsapi
import xarray

print("cdsapi:", cdsapi.__file__)
print("xarray:", xarray.__version__)
PY

mkdir -p "$OUT_DIR"

echo "Repository: $SLURM_SUBMIT_DIR"
echo "Architecture: $(uname -m)"
echo "Array index: $ARRAY_INDEX"
echo "Year: $YEAR"
echo "Output: $OUT_DIR"
echo "Python: $(command -v "$PYTHON_BIN")"

ERA5_ARGS=(
    --start "$YEAR"
    --end "$YEAR"
    --out "$OUT_DIR"
)
if [[ "${ERA5_EXTENDED:-0}" == "1" ]]; then
    ERA5_ARGS+=(--extended)
fi
if [[ "${ERA5_ENSEMBLE:-0}" == "1" ]]; then
    ERA5_ARGS+=(--ensemble)
fi

srun --ntasks=1 "$PYTHON_BIN" -u scripts/00_download_era5.py "${ERA5_ARGS[@]}"
