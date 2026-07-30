#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
mkdir -p logs

echo "Submitting real-data GH200 training preflight"
exec sbatch "$@" slurm/preflight_training_gh200.sbatch
