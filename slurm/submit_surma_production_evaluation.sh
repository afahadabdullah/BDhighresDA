#!/usr/bin/env bash
# Queue saved-output diagnostics after successful final production validation.
set -euo pipefail
TASK_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$TASK_SCRIPT_DIR/.." && pwd)"
TASK_FINAL_JOB="${1:?usage: bash slurm/submit_surma_production_evaluation.sh FINAL_VALIDATION_JOB_ID [sbatch options]}"
shift
[[ "$TASK_FINAL_JOB" =~ ^[0-9]+$ ]] || { echo 'ERROR: numeric final validation job ID required' >&2; exit 2; }
mkdir -p logs
TASK_RESULT="$(sbatch --parsable --export=ALL --dependency="afterok:$TASK_FINAL_JOB" "$@" slurm/surma_production_evaluation.sbatch)"
TASK_JOB="${TASK_RESULT%%;*}"
[[ "$TASK_JOB" =~ ^[0-9]+$ ]] || { echo "ERROR: unexpected sbatch result: $TASK_RESULT" >&2; exit 1; }
echo "Evaluation job: $TASK_JOB; waits for successful final validation: $TASK_FINAL_JOB"
echo "Log: logs/surma-prod-eval-$TASK_JOB.out"
echo "Output: ${SURMA_EVAL_OUT:-results/surma_production_2001_2024_diagnostics}"
