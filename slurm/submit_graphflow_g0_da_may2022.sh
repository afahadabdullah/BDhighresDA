#!/usr/bin/env bash
set -euo pipefail

# Submit five GraphFlow folds and summarize them against the completed CPCv2
# May 1--10, 2022 refinement folds.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

export GRAPHFLOW_DA_ROOT="${GRAPHFLOW_DA_ROOT:-runs/graphflow_g0_multimesh/da_may2022}"
export GRAPHFLOW_CPC_ROOT="${GRAPHFLOW_CPC_ROOT:-data/processed/v2_simultaneous_refinement/ing2022_s04}"
export GRAPHFLOW_IMERG="${GRAPHFLOW_IMERG:-data/processed/imerg_prepared_ing2022/imerg_0p4deg_20220501_20220510.nc}"
export GRAPHFLOW_CKPT="${GRAPHFLOW_CKPT:-runs/prior_h100_cpc_graphflow_g0/best.pt}"
export BMD_CKPT="${BMD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"

for required in "$GRAPHFLOW_CKPT" "$BMD_CKPT" "$GRAPHFLOW_IMERG"; do
    [[ -s "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done
for fold in 0 1 2 3 4; do
    for required in "$GRAPHFLOW_CPC_ROOT/fold${fold}.npz" \
                    "$GRAPHFLOW_CPC_ROOT/fold${fold}.json" \
                    "$GRAPHFLOW_CPC_ROOT/fold${fold}_bmd.csv"; do
        [[ -s "$required" ]] || {
            echo "ERROR: completed CPCv2 May 1--10 reference missing: $required" >&2
            echo "Set GRAPHFLOW_CPC_ROOT to the completed simultaneous-refinement directory." >&2
            exit 1
        }
    done
done

echo "Submitting GraphFlow G0 matched DA evaluation"
echo "  period: 2022-05-01 through 2022-05-10"
echo "  five held-out-station folds; 30 members; frozen DA settings"
echo "  GraphFlow checkpoint: $GRAPHFLOW_CKPT"
echo "  CPCv2 reference (reused): $GRAPHFLOW_CPC_ROOT"
echo "  output: $GRAPHFLOW_DA_ROOT"

array_result="$(sbatch --parsable --export=ALL "$@" slurm/graphflow_g0_da_may2022.sbatch)"
array_job="${array_result%%;*}"
summary_result="$(sbatch --parsable --dependency="afterok:${array_job}" \
    --export=ALL "$@" slurm/graphflow_g0_da_may2022_summary.sbatch)"
summary_job="${summary_result%%;*}"

echo "submitted GraphFlow fold array: $array_job"
echo "submitted dependent summary: $summary_job"
echo "monitor: squeue -u $USER"
echo "logs:    logs/graphflow-g0-da-may22-${array_job}_*.out"
echo "result:  $GRAPHFLOW_DA_ROOT/summary/comparison.{md,json,png}"
