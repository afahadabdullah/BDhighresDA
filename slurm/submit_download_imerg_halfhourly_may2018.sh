#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs
exec sbatch "$@" slurm/download_imerg_halfhourly_may2018.sbatch
