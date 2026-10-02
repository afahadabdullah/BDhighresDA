#!/usr/bin/env bash
#SBATCH --job-name=surma-prod-01-24
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
module purge 2>/dev/null || true
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/src" MPLBACKEND=Agg
unset PYTHONHOME
TASK_PYTHON="${PYTHON_BIN:-${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}/bin/python}"
[[ -x "$TASK_PYTHON" ]] || { echo "ERROR: Python not found: $TASK_PYTHON" >&2; exit 1; }
[[ "$(uname -m)" == "aarch64" ]] || { echo "ERROR: Grace/GH200 node required" >&2; exit 1; }
"$TASK_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"'
"$TASK_PYTHON" -u scripts/100_surma_production.py production \
  --start-year "${SURMA_PROD_START_YEAR:-2001}" --end-year "${SURMA_PROD_END_YEAR:-2024}" \
  --root "${SURMA_PROD_ROOT:-data/processed/brishti05_production_2001_2024}" \
  --ckpt "${SURMA_PROD_CKPT:-runs/prior_h100_cpc_v2/best.pt}" \
  --stats "${SURMA_PROD_STATS:-data/processed/stats_cpc_v2.json}" \
  --task "${SLURM_ARRAY_TASK_ID:?array required}"
