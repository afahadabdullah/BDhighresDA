#!/usr/bin/env bash
#SBATCH --job-name=imerg-daily-hh-check
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/imerg-daily-hh-check-%j.out

set -euo pipefail
module purge 2>/dev/null || true
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
export PYTHONPATH="$PWD/src"
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN"; exit 1; }
echo "CPU IMERG comparison/audit; job ${SLURM_JOB_ID:-local}; host $(hostname)"
exec "$PYTHON_BIN" -u scripts/102_compare_imerg_daily_halfhourly.py "$@"
