#!/usr/bin/env bash
set -euo pipefail

# CPU-only postprocessing: compare the existing v1 RAW/S04 folds with the
# completed corrected CPC-v2 ingestion triplet. No DA is rerun.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

export V1_RAW_ROOT="${V1_RAW_ROOT:-data/processed/bmd_imerg_eval_ing2022_RAW}"
export V1_S04_ROOT="${V1_S04_ROOT:-data/processed/bmd_imerg_eval_ing2022_S04}"
export V2_INGEST_ROOT="${V2_INGEST_ROOT:-data/processed/v2_ingestion_triplet/ing2022_s04_g010_sqrtfix}"
export V1_V2_COMPARE_ROOT="${V1_V2_COMPARE_ROOT:-$V2_INGEST_ROOT/v1_vs_v2}"

for fold in 0 1 2 3 4; do
    for required in \
        "$V1_RAW_ROOT/fold${fold}.npz" \
        "$V1_S04_ROOT/fold${fold}.npz" \
        "$V2_INGEST_ROOT/fold${fold}.npz"; do
        [[ -s "$required" ]] || { echo "ERROR: missing $required"; exit 1; }
    done
done

echo "Submitting matched v1-versus-v2 DA comparison (CPU only)"
echo "  v1 reported gauge reference: $V1_RAW_ROOT"
echo "  v1 matched S04 reference:    $V1_S04_ROOT"
echo "  corrected v2:                $V2_INGEST_ROOT"
echo "  outputs:                     $V1_V2_COMPARE_ROOT"

result="$(sbatch --parsable --export=ALL "$@" slurm/v1_v2_ingestion_comparison.sbatch)"
job="${result%%;*}"
echo "submitted comparison job: $job"
echo "inspect: logs/bdhires-v1-v2-compare-${job}.out"
echo "final comparison: $V1_V2_COMPARE_ROOT/v1_vs_v2_selection.{md,json,png}"
