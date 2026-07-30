#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PARTITION="${TEST_PARTITION:-grace}"

cd "$REPO_ROOT"
mkdir -p logs

echo "Submitting held-out best-checkpoint prediction diagnostics on ${PARTITION}"
exec sbatch --partition="$PARTITION" "$@" slurm/test_predictions.sbatch
