#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

START="${CHIRPS_START:-1981}"
END="${CHIRPS_END:-2025}"
MAX_PARALLEL="${CHIRPS_MAX_PARALLEL:-2}"
PARTITION="${CHIRPS_PARTITION:-grace-cpuonly}"

[[ "$START" =~ ^[0-9]{4}$ ]] || {
    echo "ERROR: CHIRPS_START must be a four-digit year."
    exit 2
}
[[ "$END" =~ ^[0-9]{4}$ ]] || {
    echo "ERROR: CHIRPS_END must be a four-digit year."
    exit 2
}
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: CHIRPS_MAX_PARALLEL must be a positive integer."
    exit 2
}
((START <= END)) || {
    echo "ERROR: CHIRPS_START must be less than or equal to CHIRPS_END."
    exit 2
}

cd "$REPO_ROOT"
mkdir -p logs

# Many Slurm installations require array indices to be smaller than
# MaxArraySize (often 1001), so calendar years such as 1981 cannot be used as
# indices. Submit zero-based indices and let the batch script map them back to
# years with CHIRPS_START_YEAR + SLURM_ARRAY_TASK_ID.
LAST_INDEX="$((END - START))"
if ((LAST_INDEX == 0)); then
    ARRAY_SPEC="0"
else
    ARRAY_SPEC="0-${LAST_INDEX}%${MAX_PARALLEL}"
fi

SBATCH_ARGS=(
    "--array=${ARRAY_SPEC}"
    "--export=ALL,CHIRPS_START_YEAR=${START}"
    "--partition=${PARTITION}"
)

echo "Submitting CHIRPS years ${START}-${END} as array ${ARRAY_SPEC} on ${PARTITION}"
exec sbatch "${SBATCH_ARGS[@]}" slurm/download_chirps.sbatch
