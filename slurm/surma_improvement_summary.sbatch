#!/usr/bin/env bash
#SBATCH --job-name=surma-improve-summary
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge 2>/dev/null || true
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
TASK_PYTHON="${PYTHON_BIN:-${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}/bin/python}"
[[ -x "$TASK_PYTHON" ]] || { echo "ERROR: Python not found: $TASK_PYTHON" >&2; exit 1; }
export PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/src"
unset PYTHONHOME
"$TASK_PYTHON" -u scripts/106_surma_improvement.py summarize \
    --out-dir "${SURMA_IMPROVE_OUT:-data/processed/surma_improvement_pilot}"
