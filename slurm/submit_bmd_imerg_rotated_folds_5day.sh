#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

array_result="$(sbatch --parsable "$@" slurm/bmd_imerg_rotated_folds_5day.sbatch)"
array_job="${array_result%%;*}"
summary_result="$(sbatch --parsable --dependency="afterok:${array_job}" "$@" slurm/bmd_imerg_rotated_folds_summary.sbatch)"
summary_job="${summary_result%%;*}"

echo "submitted rotated-fold GPU array: ${array_job}"
echo "submitted dependent CPU summary: ${summary_job}"
echo "final outputs: data/processed/bmd_imerg_offset_m1_rotated_summary.{json,png}"
