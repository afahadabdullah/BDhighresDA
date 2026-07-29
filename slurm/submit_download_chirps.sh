#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

START="${CHIRPS_START:-1981}"
END="${CHIRPS_END:-2025}"
MAX_PARALLEL="${CHIRPS_MAX_PARALLEL:-2}"
PARTITION="${CHIRPS_PARTITION:-}"

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

SBATCH_ARGS=("--array=${START}-${END}%${MAX_PARALLEL}")
if [[ -n "$PARTITION" ]]; then
    SBATCH_ARGS+=("--partition=$PARTITION")
fi

echo "Submitting CHIRPS years ${START}-${END} (${MAX_PARALLEL} simultaneous jobs)"
exec sbatch "${SBATCH_ARGS[@]}" slurm/download_chirps.sbatch
