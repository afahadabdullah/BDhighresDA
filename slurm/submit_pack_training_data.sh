#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PARTITION="${PACK_PARTITION:-grace-cpuonly}"

cd "$REPO_ROOT"
mkdir -p logs

echo "Submitting resumable training-data pack and alignment QC on ${PARTITION}"
exec sbatch --partition="$PARTITION" "$@" slurm/pack_training_data.sbatch
