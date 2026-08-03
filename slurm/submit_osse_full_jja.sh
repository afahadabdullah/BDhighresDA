#!/usr/bin/env bash
# Submit all 36 arm/year/month GPU chunks and a dependent merge/evaluation job.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

submission="$(sbatch "$@" slurm/osse_full_jja.sbatch)"
array_job="${submission##* }"
echo "$submission"

final_submission="$(
    sbatch \
        --dependency="afterok:$array_job" \
        --export="ALL,OSSE_FULL_ROOT=${OSSE_FULL_ROOT:-data/processed/osse_full_jja_2021_2024},OSSE_FULL_NETWORK=${OSSE_FULL_NETWORK:-bmd},OSSE_FULL_NETWORK_TAG=${OSSE_FULL_NETWORK_TAG:-bmd}" \
        slurm/finalize_osse_full_jja.sbatch
)"
echo "$final_submission"

echo
echo "GPU array:  $array_job (36 chunks, concurrency cap 6)"
echo "Days:       every JJA day in 2021-2024 (368)"
echo "Members:    ${OSSE_FULL_MEMBERS:-16} per arm"
echo "Network:    ${OSSE_FULL_NETWORK:-bmd} (${OSSE_FULL_NETWORK_TAG:-bmd})"
echo "Outputs:    ${OSSE_FULL_ROOT:-data/processed/osse_full_jja_2021_2024}"
echo "Monitor:    squeue -j $array_job"
echo "Logs:       logs/bdhires-osse-full-${array_job}_*.out"
