#!/usr/bin/env bash
# Reuse resource settings and afterok summary scheduling with the next-round runner.
set -euo pipefail
TASK_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$TASK_SCRIPT_DIR/.." && pwd)"
export SURMA_IMPROVE_OUT="${SURMA_IMPROVE_OUT:-data/processed/surma_tail_pilot}"
export SURMA_IMPROVE_SCRIPT=scripts/107_surma_tail_diagnostics.py
[[ -f "$SURMA_IMPROVE_OUT/pilot_plan.json" ]] || {
    echo 'ERROR: run python scripts/107_surma_tail_diagnostics.py prepare first' >&2; exit 2;
}
exec bash slurm/submit_surma_improvement.sh "$@"
