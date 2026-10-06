#!/usr/bin/env bash
#SBATCH --job-name=surma-prod-inputs
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
module purge 2>/dev/null || true
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/src" MPLBACKEND=Agg
unset PYTHONHOME
TASK_PYTHON="${PYTHON_BIN:-${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}/bin/python}"
[[ -x "$TASK_PYTHON" ]] || { echo "ERROR: Python not found: $TASK_PYTHON" >&2; exit 1; }
TASK_STAGE="${1:?stage required}"
TASK_ARGS=("$TASK_STAGE"
  --start-year "${SURMA_PROD_START_YEAR:-2001}" --end-year "${SURMA_PROD_END_YEAR:-2024}"
  --root "${SURMA_PROD_ROOT:-data/processed/brishti05_production_2001_2024}"
  --ckpt "${SURMA_PROD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
  --stats "${SURMA_PROD_STATS:-data/processed/stats_cpc_v2.json}"
  --bmd-wide "${SURMA_PROD_BMD_WIDE:-data/stations/Rainfall_daily_by_station_BMD_corrected.csv}"
  --bmd-stations "${SURMA_PROD_BMD_CATALOG:-data/stations/BMD_production_station_catalog.csv}"
  --bwdb "${SURMA_PROD_BWDB:-data/stations/BWDB_Rainfall_2000_2025_corrected.xlsx}"
  --imerg-raw "${SURMA_PROD_IMERG_RAW:-data/imerg_halfhourly}"
  --imerg-daily "${SURMA_PROD_IMERG_DAILY:-data/processed}")
[[ -z "${SURMA_PROD_DATA_ZARR:-}" ]] || TASK_ARGS+=(--data-zarr "$SURMA_PROD_DATA_ZARR")
case "$TASK_STAGE" in
  download-year|download-month|prepare) TASK_ARGS+=(--task "${SLURM_ARRAY_TASK_ID:?array required}");;
  audit) TASK_ARGS+=(--deep);;
  preflight|pack|finalize) ;;
  *) echo "ERROR: unsupported CPU stage $TASK_STAGE" >&2; exit 1;;
esac
"$TASK_PYTHON" -u scripts/100_surma_production.py "${TASK_ARGS[@]}"
