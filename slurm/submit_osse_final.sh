#!/usr/bin/env bash
# Submit a short exact-observation QC run, then the full matched three-arm
# overnight experiment, then the CPU-only paper summary. The full array starts
# only if the QC job succeeds.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs
SBATCH_ARGS=("$@")

FINAL_DAYS="${OSSE_FINAL_DAYS:-36}"
FINAL_MEMBERS="${OSSE_FINAL_MEMBERS:-16}"
FINAL_ROOT="${OSSE_FINAL_ROOT:-data/processed/osse_final_jja_2021_2024}"
SMOKE_ROOT="${OSSE_FINAL_SMOKE_ROOT:-data/processed/osse_final_smoke}"

smoke_submission="$(
    sbatch \
        "${SBATCH_ARGS[@]}" \
        --array=2 \
        --export="ALL,OSSE_FINAL_DAYS=3,OSSE_FINAL_MEMBERS=4,OSSE_FINAL_ROOT=$SMOKE_ROOT" \
        slurm/osse_final.sbatch
)"
smoke_job="${smoke_submission##* }"
echo "$smoke_submission"

final_submission="$(
    sbatch \
        "${SBATCH_ARGS[@]}" \
        --dependency="afterok:$smoke_job" \
        --export="ALL,OSSE_FINAL_DAYS=$FINAL_DAYS,OSSE_FINAL_MEMBERS=$FINAL_MEMBERS,OSSE_FINAL_ROOT=$FINAL_ROOT" \
        slurm/osse_final.sbatch
)"
final_job="${final_submission##* }"
echo "$final_submission"

summary_submission="$(
    sbatch \
        "${SBATCH_ARGS[@]}" \
        --dependency="afterok:$final_job" \
        --export="ALL,OSSE_FINAL_ROOT=$FINAL_ROOT" \
        slurm/summarize_osse_final.sbatch
)"
echo "$summary_submission"

echo
echo "QC job:       $smoke_job (combined, 3 days x 4 members)"
echo "Final array:  $final_job (3 arms, $FINAL_DAYS days x $FINAL_MEMBERS members)"
echo "Outputs:      $FINAL_ROOT"
echo "Monitor:      tail -f logs/bdhires-osse-final-${smoke_job}_2.out"
