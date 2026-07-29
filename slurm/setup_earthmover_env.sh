#!/usr/bin/env bash
# Create or update the ARM-native Earthmover download environment.
#
# Run this script from a Prism login node. It re-launches itself on the
# grace-cpuonly partition because the Miniforge installation and download
# environment are aarch64 binaries and cannot execute on an x86_64 login node.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

PARTITION="${ERA5_SETUP_PARTITION:-grace-cpuonly}"
CPUS="${ERA5_SETUP_CPUS:-4}"
MEMORY="${ERA5_SETUP_MEMORY:-12G}"
TIME_LIMIT="${ERA5_SETUP_TIME:-01:00:00}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Launching Earthmover environment setup on ARM partition ${PARTITION}"
    exec srun \
        --partition="$PARTITION" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$CPUS" \
        --mem="$MEMORY" \
        --time="$TIME_LIMIT" \
        "$SCRIPT_PATH"
fi

echo "Node: $(hostname)"
echo "Architecture: $(uname -m)"
[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: Earthmover setup requires an aarch64 node."
    echo "Set ERA5_SETUP_PARTITION to an ARM CPU partition."
    exit 1
}

ENV_PREFIX="${ERA5_ENV_PREFIX:-$REPO_ROOT/../envs/bdda-earthmover}"
ENV_FILE="$REPO_ROOT/environment-earthmover.yml"

if [[ -n "${BDDA_CONDA_BASE:-}" ]]; then
    CONDA_BASE_CANDIDATES=("$BDDA_CONDA_BASE")
else
    CONDA_BASE_CANDIDATES=(
        "$REPO_ROOT/../miniforge3-aarch64"
        "/home/afahad/nb/project/BDDA/miniforge3-aarch64"
        "/panfs/ccds02/nobackup/people/afahad/project/BDDA/miniforge3-aarch64"
    )
fi

CONDA_BIN=""
for base in "${CONDA_BASE_CANDIDATES[@]}"; do
    if [[ -x "$base/bin/conda" && -x "$base/bin/python" ]]; then
        CONDA_BIN="$base/bin/conda"
        break
    fi
done

[[ -n "$CONDA_BIN" ]] || {
    echo "ERROR: ARM Miniforge was not found."
    echo "Set BDDA_CONDA_BASE=/absolute/path/to/miniforge3-aarch64."
    exit 1
}

echo "Conda: $CONDA_BIN"
echo "Environment: $ENV_PREFIX"
"$CONDA_BIN" --version

if [[ -d "$ENV_PREFIX/conda-meta" ]]; then
    echo "Updating existing Earthmover environment"
    "$CONDA_BIN" env update \
        --prefix "$ENV_PREFIX" \
        --file "$ENV_FILE" \
        --prune
else
    echo "Creating Earthmover environment"
    "$CONDA_BIN" env create \
        --prefix "$ENV_PREFIX" \
        --file "$ENV_FILE"
fi

export PYTHONNOUSERSITE=1
"$ENV_PREFIX/bin/python" - <<'PY'
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
print("Earthmover environment OK")
PY
