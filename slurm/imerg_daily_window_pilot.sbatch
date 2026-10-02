#!/usr/bin/env bash
#SBATCH --job-name=imerg-daily-pilot
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

# Submit from the repository root after mkdir -p logs. Arguments after this
# filename are forwarded to the five-day, one-member comparison script:
# sbatch slurm/imerg_daily_window_pilot.sbatch --download-imerg
set -euo pipefail
module purge 2>/dev/null || true
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/src" MPLBACKEND=Agg
unset PYTHONHOME
TASK_PYTHON="${PYTHON_BIN:-${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}/bin/python}"
[[ -x "$TASK_PYTHON" ]] || { echo "ERROR: Python not found: $TASK_PYTHON" >&2; exit 1; }
[[ "$(uname -m)" == "aarch64" ]] || { echo "ERROR: Grace/GH200 node required" >&2; exit 1; }
"$TASK_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"'

echo "IMERG timing pilot: 1–5 May 2022 gauge reports; one member/day/case"
echo "Job: ${SLURM_JOB_ID:-unknown}; host: $(hostname); Python: $TASK_PYTHON"
"$TASK_PYTHON" -u scripts/101_compare_imerg_daily_windows.py "$@"
