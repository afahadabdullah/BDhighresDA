#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
ARRAY_JOB=$(sbatch --parsable slurm/osse_footprint_ablation.sbatch)
FINAL_JOB=$(sbatch --parsable --dependency="afterok:${ARRAY_JOB}" \
    slurm/finalize_osse_footprint_ablation.sbatch)

echo "submitted footprint-ablation array: $ARRAY_JOB"
echo "submitted dependent finalizer:       $FINAL_JOB"
echo "monitor: squeue -j ${ARRAY_JOB},${FINAL_JOB}"
