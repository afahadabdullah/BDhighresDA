#!/usr/bin/env bash
#SBATCH --job-name=surma-paper1-eval
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

# Post-process saved test-year ensembles; no model training or generation.
set -euo pipefail
module purge 2>/dev/null || true
TASK_ENV="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$TASK_ENV/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN" >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg
unset PYTHONHOME PYTHONPATH
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/surma-paper1-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" logs
TASK_ARGS=(--profile "${PAPER1_PROFILE:-superob-final}"
           --out-dir "${PAPER1_OUT:-output/paper1_cpcv2}")
[[ -z "${PAPER1_ROOT:-}" ]] || TASK_ARGS+=(--root "$PAPER1_ROOT")
[[ "${PAPER1_GRIDDED:-1}" != "1" ]] || TASK_ARGS+=(--with-gridded)
[[ -z "${PAPER1_CPC_SOURCE:-}" ]] || TASK_ARGS+=(--cpc-source-zarr "$PAPER1_CPC_SOURCE")
"$PYTHON_BIN" -u scripts/90_evaluate_cpcv2_paper1.py "${TASK_ARGS[@]}"
