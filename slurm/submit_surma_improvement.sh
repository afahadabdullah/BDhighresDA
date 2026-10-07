#!/usr/bin/env bash
# Submit an already prepared research pilot; all data remain in ignored paths.
set -euo pipefail
TASK_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$TASK_SCRIPT_DIR/.." && pwd)"
TASK_OUT="${SURMA_IMPROVE_OUT:-data/processed/surma_improvement_pilot}"
TASK_CONCURRENCY="${SURMA_IMPROVE_CONCURRENCY:-2}"
TASK_PYTHON="${PYTHON_BIN:-python3}"
[[ "$TASK_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || { echo 'ERROR: concurrency must be positive integer' >&2; exit 2; }
[[ -f "$TASK_OUT/pilot_plan.json" ]] || { echo "ERROR: run scripts/106_surma_improvement.py prepare first: $TASK_OUT/pilot_plan.json missing" >&2; exit 2; }
TASK_COUNT="$("$TASK_PYTHON" - "$TASK_OUT/pilot_plan.json" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
assert plan['status'] == 'prepared_research_pilot' and len(plan['windows']) > 0
print(len(plan['windows']))
PY
)"
mkdir -p logs
export SURMA_IMPROVE_OUT="$TASK_OUT"
TASK_RAW="$(sbatch --parsable --export=ALL --array="0-$((TASK_COUNT-1))%$TASK_CONCURRENCY" slurm/surma_improvement.sbatch)"
TASK_ARRAY="${TASK_RAW%%;*}"
[[ "$TASK_ARRAY" =~ ^[0-9]+$ ]] || { echo "ERROR: unexpected sbatch result: $TASK_RAW" >&2; exit 1; }
echo "GPU pilot array: $TASK_ARRAY; $TASK_COUNT windows; concurrency $TASK_CONCURRENCY"
TASK_RAW="$(sbatch --parsable --export=ALL --dependency="afterok:$TASK_ARRAY" slurm/surma_improvement_summary.sbatch)"
TASK_SUMMARY="${TASK_RAW%%;*}"
[[ "$TASK_SUMMARY" =~ ^[0-9]+$ ]] || { echo "ERROR: unexpected sbatch result: $TASK_RAW" >&2; exit 1; }
echo "Dependent summary: $TASK_SUMMARY"
echo "Results: $TASK_OUT/summary/comparison.md"
