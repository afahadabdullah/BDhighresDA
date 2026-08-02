#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

START_YEAR="${IMERG_START_YEAR:-2021}"
END_YEAR="${IMERG_END_YEAR:-2024}"
MAX_ACTIVE="${IMERG_ARRAY_CONCURRENCY:-2}"
if (( END_YEAR < START_YEAR )); then
    echo "ERROR: IMERG_END_YEAR must not precede IMERG_START_YEAR" >&2
    exit 1
fi
if (( MAX_ACTIVE < 1 )); then
    echo "ERROR: IMERG_ARRAY_CONCURRENCY must be positive" >&2
    exit 1
fi

TASK_COUNT=$(((END_YEAR - START_YEAR + 1) * 12))
ARRAY_SPEC="0-$((TASK_COUNT - 1))%$MAX_ACTIVE"
echo "Submitting $TASK_COUNT monthly IMERG tasks for $START_YEAR-$END_YEAR (max $MAX_ACTIVE active)"
exec sbatch \
    --array="$ARRAY_SPEC" \
    --export="ALL,IMERG_START_YEAR=$START_YEAR,IMERG_END_YEAR=$END_YEAR" \
    "$@" \
    slurm/download_imerg_halfhourly_2021_2024.sbatch
