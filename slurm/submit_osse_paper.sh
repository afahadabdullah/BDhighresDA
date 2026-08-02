#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

submission="$(sbatch "$@" slurm/osse_paper.sbatch)"
job_id="${submission##* }"
echo "$submission"
echo "Submitting summary after array job $job_id"
sbatch --dependency="afterok:$job_id" slurm/summarize_osse_paper.sbatch
