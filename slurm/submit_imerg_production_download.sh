#!/usr/bin/env bash
# Submit one coordinator, rather than multiplying NASA connections in an array.
set -euo pipefail
TASK_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_SCRIPT_DIR/.."
TASK_EXTRA=(--export=ALL)
TASK_DRY=0
TASK_ARGS=(--start-year "${IMERG_START_YEAR:-2001}" --end-year "${IMERG_END_YEAR:-2024}"
           --connections "${IMERG_DOWNLOAD_CONNECTIONS:-3}"
           --raw "${SURMA_PROD_IMERG_RAW:-data/imerg_halfhourly}"
           --daily "${SURMA_PROD_IMERG_DAILY:-data/processed}"
           --state "${IMERG_DOWNLOAD_STATE:-data/processed/imerg_download_2001_2024}")
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) TASK_DRY=1;;
        --account=*|--time=*) TASK_EXTRA+=("$1");;
        --transport=auto|--transport=requests|--transport=wget) TASK_ARGS+=(--transport "${1#--transport=}");;
        --month=*)
            TASK_MONTH="${1#--month=}"
            [[ "$TASK_MONTH" =~ ^[0-9]{4}-(0[1-9]|1[0-2])$ ]] || { echo "ERROR: month must be YYYY-MM" >&2; exit 2; }
            [[ "${IMERG_START_YEAR:-2001}" =~ ^[0-9]{4}$ && "${IMERG_END_YEAR:-2024}" =~ ^[0-9]{4}$ ]] || { echo "ERROR: four-digit years required" >&2; exit 2; }
            (( ${TASK_MONTH:0:4} >= ${IMERG_START_YEAR:-2001} && ${TASK_MONTH:0:4} <= ${IMERG_END_YEAR:-2024} )) || { echo "ERROR: month is outside the year range" >&2; exit 2; }
            TASK_ARGS+=(--month "$TASK_MONTH");;
        *) echo "ERROR: unsupported option $1; use --dry-run, --account=NAME, --time=HH:MM:SS, --month=YYYY-MM, --transport=auto|requests|wget" >&2; exit 2;;
    esac
    shift
done
[[ "${IMERG_START_YEAR:-2001}" =~ ^[0-9]{4}$ && "${IMERG_END_YEAR:-2024}" =~ ^[0-9]{4}$ ]] || { echo "ERROR: four-digit years required" >&2; exit 2; }
(( ${IMERG_START_YEAR:-2001} >= 2001 && ${IMERG_START_YEAR:-2001} <= ${IMERG_END_YEAR:-2024} && ${IMERG_END_YEAR:-2024} <= 2024 )) || { echo "ERROR: years must be within 2001..2024" >&2; exit 2; }
[[ "${IMERG_DOWNLOAD_CONNECTIONS:-3}" =~ ^[123]$ ]] || { echo "ERROR: connections must be 1, 2 or 3" >&2; exit 2; }
if (( TASK_DRY )); then
    printf 'sbatch'; printf ' %q' --parsable "${TASK_EXTRA[@]}" slurm/imerg_production_download.sbatch "${TASK_ARGS[@]}"; printf '\n'
    exit 0
fi
mkdir -p logs
TASK_JOB="$(sbatch --parsable "${TASK_EXTRA[@]}" slurm/imerg_production_download.sbatch "${TASK_ARGS[@]}")"
TASK_JOB="${TASK_JOB%%;*}"
[[ "$TASK_JOB" =~ ^[0-9]+$ ]] || { echo "ERROR: unexpected sbatch result: $TASK_JOB" >&2; exit 1; }
echo "Download job: $TASK_JOB"
echo "Follow: tail -f logs/imerg-hh-production-$TASK_JOB.out"
echo "Progress: ${IMERG_DOWNLOAD_STATE:-data/processed/imerg_download_2001_2024}/status.json"
echo "IMERG ready: ${IMERG_DOWNLOAD_STATE:-data/processed/imerg_download_2001_2024}/IMERG_READY.json"
echo "Resume after a timeout/failure: rerun this submission with the same options."
