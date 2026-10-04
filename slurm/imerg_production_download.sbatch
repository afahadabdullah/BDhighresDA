#!/usr/bin/env bash
#SBATCH --job-name=imerg-hh-production
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/imerg-hh-production-%j.out

set -euo pipefail
module purge 2>/dev/null || true
cd "$SLURM_SUBMIT_DIR"
TASK_ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
TASK_PYTHON="${IMERG_PYTHON:-$TASK_ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
export PYTHONPATH="$PWD/src"
[[ -x "$TASK_PYTHON" ]] || { echo "ERROR: Python missing: $TASK_PYTHON" >&2; exit 1; }
[[ -f "$HOME/.netrc" ]] || { echo "ERROR: Earthdata ~/.netrc missing" >&2; exit 1; }
"$TASK_PYTHON" -c 'import numpy, xarray, netCDF4'
echo "IMERG half-hourly production download; job $SLURM_JOB_ID; host $(hostname)"
exec "$TASK_PYTHON" -u scripts/103_download_imerg_production.py "$@"
