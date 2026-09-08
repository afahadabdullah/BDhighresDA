#!/usr/bin/env bash
set -euo pipefail

# Submit the broad GraphFlow DA screen on one spatial fold.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

export GRAPHFLOW_DA_SCREEN_FOLD="${GRAPHFLOW_DA_SCREEN_FOLD:-0}"
export GRAPHFLOW_DA_SCREEN_ROOT="${GRAPHFLOW_DA_SCREEN_ROOT:-runs/graphflow_g0_multimesh/da_screen_may2022}"
export GRAPHFLOW_CPC_ROOT="${GRAPHFLOW_CPC_ROOT:-data/processed/v2_simultaneous_refinement/ing2022_s04}"
export GRAPHFLOW_IMERG="${GRAPHFLOW_IMERG:-data/processed/imerg_prepared_ing2022/imerg_0p4deg_20220501_20220510.nc}"
export GRAPHFLOW_CKPT="${GRAPHFLOW_CKPT:-runs/prior_h100_cpc_graphflow_g0/best.pt}"
export BMD_CKPT="${BMD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"

for required in "$GRAPHFLOW_CKPT" "$BMD_CKPT" "$GRAPHFLOW_IMERG" \
                "$GRAPHFLOW_CPC_ROOT/fold${GRAPHFLOW_DA_SCREEN_FOLD}_bmd.csv"; do
    [[ -s "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done

echo "Submitting broad GraphFlow DA method screen"
echo "  period: 2022-05-01 through 2022-05-10"
echo "  fold: $((GRAPHFLOW_DA_SCREEN_FOLD + 1))/5; members: 30"
echo "  output: $GRAPHFLOW_DA_SCREEN_ROOT"
echo "  this is a screening run; verify only the shortlist on all five folds"

submit_result="$(sbatch --parsable --export=ALL "$@" slurm/graphflow_g0_da_screen_may2022.sbatch)"
job_id="${submit_result%%;*}"

echo "submitted GPU job: $job_id"
echo "monitor: squeue -j $job_id"
echo "log:     logs/graphflow-g0-da-screen-${job_id}.out"
echo "result:  $GRAPHFLOW_DA_SCREEN_ROOT/fold${GRAPHFLOW_DA_SCREEN_FOLD}_screen.{md,json,png}"
