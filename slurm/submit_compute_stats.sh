#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PARTITION="${STATS_PARTITION:-grace-cpuonly}"

cd "$REPO_ROOT"
mkdir -p logs

echo "Submitting training-period normalization statistics on ${PARTITION}"
exec sbatch --partition="$PARTITION" "$@" slurm/compute_stats.sbatch
