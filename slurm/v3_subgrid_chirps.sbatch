#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v3-chirps
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: ARM Python not found: $PYTHON_BIN"; exit 1; }
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD/src"
mkdir -p logs

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: this job requires an aarch64 Grace node" >&2
    exit 1
}
[[ -f scripts/01_download_chirps.py ]] || {
    echo "ERROR: submit from the BDhighresDA repository root" >&2
    exit 1
}

START="${V3_CHIRPS_START:-1981}"
END="${V3_CHIRPS_END:-2024}"
TASK="${SLURM_ARRAY_TASK_ID:?submit as an array through submit_v3_subgrid_inputs.sh}"
YEAR=$((START + TASK))
OUT="${V3_CHIRPS_OUT:-data/raw/chirps_v3sg}"
if (( YEAR < START || YEAR > END )); then
    echo "ERROR: task $TASK maps to $YEAR outside $START..$END" >&2
    exit 1
fi

echo "Preparing aligned V3-SG CHIRPS year $YEAR"
echo "  output directory: $OUT"
"$PYTHON_BIN" -u scripts/01_download_chirps.py \
    --start "$YEAR" --end "$YEAR" --grid wide_cpc --out "$OUT"

TARGET="$OUT/chirps_wide_cpc_${YEAR}.nc"
"$PYTHON_BIN" - "$TARGET" "$YEAR" <<'PY'
import calendar
import sys

import numpy as np
import xarray as xr

path, year_text = sys.argv[1:]
year = int(year_text)
with xr.open_dataset(path) as dataset:
    field = dataset["precip"]
    assert field.sizes["latitude"] == 260
    assert field.sizes["longitude"] == 260
    assert field.sizes["time"] == (366 if calendar.isleap(year) else 365)
    assert np.all(np.diff(field.latitude.values) > 0)
    assert np.all(np.diff(field.longitude.values) > 0)
print("Validated:", path)
PY

