#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
mkdir -p logs

echo "Submitting one-time PyTorch setup in the existing bdda-gh200 environment"
exec sbatch "$@" slurm/setup_pytorch_gh200.sbatch
