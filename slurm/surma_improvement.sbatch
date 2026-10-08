#!/usr/bin/env bash
#SBATCH --job-name=surma-improve
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
module purge 2>/dev/null || true
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
TASK_PYTHON="${PYTHON_BIN:-${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}/bin/python}"
[[ -x "$TASK_PYTHON" ]] || { echo "ERROR: Python not found: $TASK_PYTHON" >&2; exit 1; }
[[ "$(uname -m)" == aarch64 ]] || { echo 'ERROR: Grace/GH200 aarch64 node required' >&2; exit 1; }
export PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/src" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONHOME
"$TASK_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
"$TASK_PYTHON" -u "${SURMA_IMPROVE_SCRIPT:-scripts/106_surma_improvement.py}" run \
    --out-dir "${SURMA_IMPROVE_OUT:-data/processed/surma_improvement_pilot}" \
    --task "${SLURM_ARRAY_TASK_ID:?array task required}"
