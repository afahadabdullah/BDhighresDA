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

START_YEAR="${CHIRPS_START_YEAR:?Use slurm/submit_download_chirps.sh to submit this job.}"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?Submit this file as a Slurm array job.}"
YEAR="$((START_YEAR + ARRAY_INDEX))"
OUT_DIR="${CHIRPS_OUT:-data/raw/chirps}"

export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

if [[ -n "${CHIRPS_PYTHON:-}" ]]; then
    PYTHON_BIN="$CHIRPS_PYTHON"
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
        echo "Set CHIRPS_PYTHON=/absolute/path/to/ARM/python and resubmit."
        exit 1
    }
else
    PYTHON_BIN="$(command -v python || command -v python3 || true)"
fi

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
echo "Architecture: $(uname -m)"
echo "Array index: $ARRAY_INDEX"
echo "Year: $YEAR"
echo "Output: $OUT_DIR"
echo "Python: $(command -v "$PYTHON_BIN")"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/01_download_chirps.py \
    --start "$YEAR" \
    --end "$YEAR" \
    --out "$OUT_DIR"
