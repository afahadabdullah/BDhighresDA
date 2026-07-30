#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PARTITION="${NORM_PARTITION:-grace-cpuonly}"

cd "$REPO_ROOT"
mkdir -p logs

echo "Submitting normalization diagnostic figure on ${PARTITION}"
exec sbatch --partition="$PARTITION" "$@" slurm/normalization_diagnostics.sbatch
