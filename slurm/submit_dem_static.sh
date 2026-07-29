#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PARTITION="${DEM_PARTITION:-grace-cpuonly}"

cd "$REPO_ROOT"
mkdir -p logs

echo "Submitting Copernicus DEM download and static-field build on ${PARTITION}"
exec sbatch --partition="$PARTITION" "$@" slurm/download_dem_static.sbatch
