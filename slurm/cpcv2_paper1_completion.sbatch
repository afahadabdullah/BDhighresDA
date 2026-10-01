#!/usr/bin/env bash
#SBATCH --job-name=surma-paper1-complete
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

# Fill available manuscript slots from actual saved data; no inference.
set -euo pipefail
module purge 2>/dev/null || true
TASK_ENV="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$TASK_ENV/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN" >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg
unset PYTHONHOME PYTHONPATH
export MPLCONFIGDIR="${TMPDIR:-/tmp}/surma-paper1-completion-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
TASK_ARGS=(--root "${PAPER1_ROOT:-data/processed/v2_bmd_bwdb_superob_2021_2024}"
           --output "${PAPER1_OUTPUT:-manuscript/additional}"
           --history "${PAPER1_HISTORY:-runs/prior_h100_cpc_v2/validation/history.jsonl}")
[[ -z "${PAPER1_SELECTION:-}" ]] || TASK_ARGS+=(--selection "$PAPER1_SELECTION")
[[ -z "${PAPER1_COMPUTE:-}" ]] || TASK_ARGS+=(--compute "$PAPER1_COMPUTE")
"$PYTHON_BIN" -u scripts/92_complete_cpcv2_paper1.py "${TASK_ARGS[@]}"
