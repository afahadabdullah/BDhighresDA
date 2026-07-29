#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

START="${ERA5_START:-1981}"
END="${ERA5_END:-2025}"
MAX_PARALLEL="${ERA5_MAX_PARALLEL:-2}"
PARTITION="${ERA5_PARTITION:-grace-cpuonly}"

[[ "$START" =~ ^[0-9]{4}$ ]] || {
    echo "ERROR: ERA5_START must be a four-digit year."
    exit 2
}
[[ "$END" =~ ^[0-9]{4}$ ]] || {
    echo "ERROR: ERA5_END must be a four-digit year."
    exit 2
}
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: ERA5_MAX_PARALLEL must be a positive integer."
    exit 2
}
((START <= END)) || {
    echo "ERROR: ERA5_START must be less than or equal to ERA5_END."
    exit 2
}

cd "$REPO_ROOT"
mkdir -p logs

LAST_INDEX="$((END - START))"
if ((LAST_INDEX == 0)); then
    ARRAY_SPEC="0"
else
    ARRAY_SPEC="0-${LAST_INDEX}%${MAX_PARALLEL}"
fi

SBATCH_ARGS=(
    "--array=${ARRAY_SPEC}"
    "--export=ALL,ERA5_START_YEAR=${START}"
    "--partition=${PARTITION}"
)

echo "Submitting ERA5 years ${START}-${END} as array ${ARRAY_SPEC} on ${PARTITION}"
exec sbatch "${SBATCH_ARGS[@]}" slurm/download_era5.sbatch
