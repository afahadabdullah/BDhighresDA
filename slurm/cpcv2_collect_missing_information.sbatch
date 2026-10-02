#!/usr/bin/env bash
#SBATCH --job-name=surma-paper-collect
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge 2>/dev/null || true
TASK_ENV="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$TASK_ENV/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN" >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg
unset PYTHONHOME PYTHONPATH
TASK_ARGS=(--prepare-baseline-inputs
           --root "${PAPER1_ROOT:-data/processed/v2_bmd_bwdb_superob_2021_2024}"
           --out-dir "${PAPER1_COLLECTION_OUT:-output/paper1_information_${SLURM_JOB_ID}}")
# Omission uses the collector's completed-v2 discovery on this host.
[[ -z "${PAPER1_EVIDENCE_DIR:-}" ]] || TASK_ARGS+=(--evidence-dir "$PAPER1_EVIDENCE_DIR")
[[ -z "${PAPER1_SELECTION_ROOT:-}" ]] || TASK_ARGS+=(--selection-root "$PAPER1_SELECTION_ROOT")
if [[ -n "${PAPER1_LOG_LIST:-}" ]]; then
    mapfile -t TASK_LOGS < "$PAPER1_LOG_LIST"
    [[ ${#TASK_LOGS[@]} -gt 0 ]] || { echo "ERROR: empty log list" >&2; exit 1; }
    TASK_ARGS+=(--logs "${TASK_LOGS[@]}")
fi
if [[ -n "${PAPER1_JOB_ID_LIST:-}" ]]; then
    mapfile -t TASK_JOBS < "$PAPER1_JOB_ID_LIST"
    [[ ${#TASK_JOBS[@]} -gt 0 ]] || { echo "ERROR: empty job ID list" >&2; exit 1; }
    TASK_ARGS+=(--accounting-job-ids "${TASK_JOBS[@]}")
fi
"$PYTHON_BIN" -u scripts/98_collect_paper1_missing_information.py "${TASK_ARGS[@]}"
