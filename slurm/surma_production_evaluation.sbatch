#!/usr/bin/env bash
#SBATCH --job-name=surma-prod-eval
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge 2>/dev/null || true
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
TASK_PYTHON="${PYTHON_BIN:-${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}/bin/python}"
[[ -x "$TASK_PYTHON" ]] || { echo "ERROR: Python not found: $TASK_PYTHON" >&2; exit 1; }
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg PYTHONPATH="$PWD/src"
unset PYTHONHOME
export MPLCONFIGDIR="${TMPDIR:-/tmp}/surma-production-eval-${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$MPLCONFIGDIR"
"$TASK_PYTHON" -c 'import numpy, pandas, matplotlib, zarr, numcodecs'
TASK_ARGS=(--root "${SURMA_PROD_ROOT:-data/processed/brishti05_production_2001_2024}"
           --out-dir "${SURMA_EVAL_OUT:-results/surma_production_2001_2024_diagnostics}"
           --boundary-geojson "${SURMA_EVAL_BOUNDARY:-configs/geography/geoBoundaries-BGD-ADM0.geojson}")
[[ -z "${SURMA_EVAL_START_YEAR:-}" ]] || TASK_ARGS+=(--start-year "$SURMA_EVAL_START_YEAR")
[[ -z "${SURMA_EVAL_END_YEAR:-}" ]] || TASK_ARGS+=(--end-year "$SURMA_EVAL_END_YEAR")
"$TASK_PYTHON" -u scripts/105_evaluate_surma_production.py "${TASK_ARGS[@]}"
