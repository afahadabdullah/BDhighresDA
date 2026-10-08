#!/usr/bin/env bash
set -euo pipefail
TASK_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$TASK_SCRIPT_DIR/.." && pwd)"
export SURMA_IMPROVE_OUT="${SURMA_IMPROVE_OUT:-data/processed/surma_event_attribution}"
export SURMA_IMPROVE_SCRIPT=scripts/108_surma_event_attribution.py
[[ -f "$SURMA_IMPROVE_OUT/pilot_plan.json" ]] || {
    echo 'ERROR: run python scripts/108_surma_event_attribution.py prepare first' >&2; exit 2;
}
exec bash slurm/submit_surma_improvement.sh "$@"
