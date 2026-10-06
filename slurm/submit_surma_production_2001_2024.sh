#!/usr/bin/env bash
# Dependency chain: download -> validate/pack -> prepare -> sample -> final audit.
# --audit-only checks PRISM inputs; --prepare-only omits GPU sampling.
# --dry-run prints the commands without submitting jobs.
set -euo pipefail
TASK_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$TASK_SCRIPT_DIR/.." && pwd)"
mkdir -p logs
export SURMA_PROD_START_YEAR="${SURMA_PROD_START_YEAR:-2001}"
export SURMA_PROD_END_YEAR="${SURMA_PROD_END_YEAR:-2024}"
export SURMA_PROD_ROOT="${SURMA_PROD_ROOT:-data/processed/brishti05_production_2001_2024}"
export SURMA_PROD_ALLOW_KNOWN_CPC_GAPS="${SURMA_PROD_ALLOW_KNOWN_CPC_GAPS:-0}"
TASK_INPUT_CONCURRENCY="${SURMA_PROD_INPUT_CONCURRENCY:-2}"
TASK_GPU_CONCURRENCY="${SURMA_PROD_CONCURRENCY:-2}"
TASK_MODE=full TASK_DRY=0 TASK_EXTRA=(--export=ALL)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --audit-only) TASK_MODE=audit;;
    --prepare-only) TASK_MODE=prepare;;
    --dry-run) TASK_DRY=1;;
    --account=*) TASK_EXTRA+=("$1");;
    *) echo "ERROR: unsupported option $1 (use --account=NAME, --audit-only, --prepare-only or --dry-run)" >&2; exit 2;;
  esac
  shift
done
[[ "$SURMA_PROD_START_YEAR" =~ ^[0-9]{4}$ && "$SURMA_PROD_END_YEAR" =~ ^[0-9]{4}$ ]] || { echo "ERROR: four-digit years required" >&2; exit 2; }
[[ "$TASK_INPUT_CONCURRENCY" =~ ^[1-9][0-9]*$ && "$TASK_GPU_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: positive concurrency required" >&2; exit 2; }
[[ "$SURMA_PROD_ALLOW_KNOWN_CPC_GAPS" =~ ^[01]$ ]] || { echo "ERROR: SURMA_PROD_ALLOW_KNOWN_CPC_GAPS must be 0 or 1" >&2; exit 2; }
(( SURMA_PROD_START_YEAR >= 2001 && SURMA_PROD_START_YEAR <= SURMA_PROD_END_YEAR && SURMA_PROD_END_YEAR <= 2024 )) || { echo "ERROR: years must be within 2001..2024" >&2; exit 2; }
TASK_YEARS=$((SURMA_PROD_END_YEAR-SURMA_PROD_START_YEAR+1))
submit() {
  if (( TASK_DRY )); then
    printf 'sbatch'; printf ' %q' --parsable "${TASK_EXTRA[@]}" "$@"; printf '\n'
  else
    sbatch --parsable "${TASK_EXTRA[@]}" "$@"
  fi
}
job_id() {
  local task_result
  task_result="$(submit "$@")" || return $?
  if (( TASK_DRY )); then
    echo "$task_result" >&2
    echo DRY_JOB
  else
    task_result="${task_result%%;*}"
    [[ "$task_result" =~ ^[0-9]+$ ]] || { echo "ERROR: unexpected sbatch result: $task_result" >&2; exit 1; }
    echo "$task_result"
  fi
}
if [[ "$TASK_MODE" == audit ]]; then
  TASK_AUDIT_JOB="$(job_id slurm/surma_production_inputs.sbatch audit)"
  echo "Input audit: $TASK_AUDIT_JOB; report: $SURMA_PROD_ROOT/input_inventory.json"
  exit 0
fi
if (( ! TASK_DRY )); then
  # Model and national gauge sources are existing project artifacts; downloads
  # cannot recreate the evaluated weights, normalization or private observations.
  for task_required in "${SURMA_PROD_CKPT:-runs/prior_h100_cpc_v2/best.pt}" \
    "${SURMA_PROD_STATS:-data/processed/stats_cpc_v2.json}" \
    "${SURMA_PROD_BMD_WIDE:-data/stations/Rainfall_daily_by_station_BMD_corrected.csv}" \
    "${SURMA_PROD_BMD_CATALOG:-data/stations/BMD_production_station_catalog.csv}" \
    "${SURMA_PROD_BWDB:-data/stations/BWDB_Rainfall_2000_2025_corrected.xlsx}"; do
    [[ -s "$task_required" ]] || { echo "ERROR: required project artifact missing: $task_required" >&2; exit 1; }
  done
fi
TASK_CHECK_JOB="$(job_id slurm/surma_production_inputs.sbatch preflight)"
TASK_ANNUAL_JOB="$(job_id --dependency="afterok:$TASK_CHECK_JOB" --array="0-${TASK_YEARS}%${TASK_INPUT_CONCURRENCY}" slurm/surma_production_inputs.sbatch download-year)"
TASK_IMERG_JOB="$(job_id --dependency="afterok:$TASK_CHECK_JOB" --array="0-$((TASK_YEARS*12-1))%${TASK_INPUT_CONCURRENCY}" slurm/surma_production_inputs.sbatch download-month)"
TASK_PACK_JOB="$(job_id --dependency="afterok:$TASK_ANNUAL_JOB" slurm/surma_production_inputs.sbatch pack)"
TASK_PREP_JOB="$(job_id --dependency="afterok:$TASK_PACK_JOB:$TASK_IMERG_JOB" --array="0-$((TASK_YEARS*4-1))%${TASK_INPUT_CONCURRENCY}" slurm/surma_production_inputs.sbatch prepare)"
echo "Preflight: $TASK_CHECK_JOB; annual inputs: $TASK_ANNUAL_JOB; IMERG months: $TASK_IMERG_JOB; predictors: $TASK_PACK_JOB; preparation: $TASK_PREP_JOB"
if [[ "$TASK_MODE" == prepare ]]; then
  echo "Preparation only. Submit this wrapper again when ready to sample; valid downloaded/packed inputs are reused."
  exit 0
fi
TASK_GPU_JOB="$(job_id --dependency="afterok:$TASK_PREP_JOB" --array="0-$((TASK_YEARS*4-1))%${TASK_GPU_CONCURRENCY}" slurm/surma_production_2001_2024.sbatch)"
TASK_FINAL_JOB="$(job_id --dependency="afterok:$TASK_GPU_JOB" slurm/surma_production_inputs.sbatch finalize)"
echo "Production: $TASK_GPU_JOB; final validation: $TASK_FINAL_JOB"
echo "Output: $SURMA_PROD_ROOT/gridded/YEAR_qN.zarr; final receipt: $SURMA_PROD_ROOT/production_manifest.json"
