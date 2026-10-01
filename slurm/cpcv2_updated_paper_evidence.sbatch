#!/usr/bin/env bash
#SBATCH --job-name=surma-paper-evidence
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

# Entire available test archive; saved-array post-processing only.
set -euo pipefail
module purge 2>/dev/null || true
TASK_ENV="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$TASK_ENV/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN" >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg
unset PYTHONHOME PYTHONPATH
export MPLCONFIGDIR="${TMPDIR:-/tmp}/surma-paper-evidence-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" logs
TASK_ARGS=(--run --run-existing
           --root "${PAPER1_ROOT:-data/processed/v2_bmd_bwdb_superob_2021_2024}"
           --out-dir "${PAPER1_EVIDENCE_OUT:-output/paper1_updated_evidence}"
           --history "${PAPER1_HISTORY:-runs/prior_h100_cpc_v2/validation/history.jsonl}")
[[ -z "${PAPER1_CPC_SOURCE:-}" ]] || TASK_ARGS+=(--cpc-source-zarr "$PAPER1_CPC_SOURCE")
[[ -z "${PAPER1_SELECTION:-}" ]] || TASK_ARGS+=(--selection "$PAPER1_SELECTION")
[[ -z "${PAPER1_COMPUTE:-}" ]] || TASK_ARGS+=(--compute "$PAPER1_COMPUTE")
[[ -z "${PAPER1_CHECKPOINT:-}" ]] || TASK_ARGS+=(--checkpoint "$PAPER1_CHECKPOINT")
[[ -z "${PAPER1_UPSTREAM_INVENTORIES:-}" ]] || TASK_ARGS+=(--upstream-inventories "$PAPER1_UPSTREAM_INVENTORIES")
if [[ -n "${PAPER1_NATIVE_IMERG_LIST:-}" ]]; then
    mapfile -t TASK_NATIVE_FILES < "$PAPER1_NATIVE_IMERG_LIST"
    [[ ${#TASK_NATIVE_FILES[@]} -gt 0 ]] || { echo "ERROR: native IMERG file list is empty" >&2; exit 1; }
    TASK_ARGS+=(--native-imerg "${TASK_NATIVE_FILES[@]}")
fi
[[ "${PAPER1_FULL_GRIDDED:-1}" != "1" ]] || TASK_ARGS+=(--with-gridded)
"$PYTHON_BIN" -u scripts/96_complete_updated_paper_evidence.py "${TASK_ARGS[@]}"
