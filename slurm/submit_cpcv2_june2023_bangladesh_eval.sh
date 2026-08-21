#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

export V2_CONFIRM_ROOT="${V2_CONFIRM_ROOT:-data/processed/v2_confirmatory_2021_2024}"
export V2_JUNE_BGD_OUT="${V2_JUNE_BGD_OUT:-$V2_CONFIRM_ROOT/evaluation/june2023_bangladesh_ig010}"
export V2_JUNE_BGD_BOUNDARY="${V2_JUNE_BGD_BOUNDARY:-$V2_JUNE_BGD_OUT/inputs/geoBoundaries-BGD-ADM0.geojson}"

PREP_PYTHON="${V2_JUNE_BGD_PREP_PYTHON:-$(command -v python3 || command -v python)}"
[[ -n "$PREP_PYTHON" && -x "$PREP_PYTHON" ]] || {
    echo "ERROR: no login-node Python found; set V2_JUNE_BGD_PREP_PYTHON" >&2
    exit 1
}

if [[ ! -s "$V2_JUNE_BGD_BOUNDARY" ]]; then
    echo "Downloading and snapshotting the published Bangladesh ADM0 boundary"
    "$PREP_PYTHON" scripts/80_fetch_bangladesh_boundary.py \
        --output "$V2_JUNE_BGD_BOUNDARY"
fi

result="$(sbatch --parsable \
    --export="ALL,V2_CONFIRM_ROOT=$V2_CONFIRM_ROOT,V2_JUNE_BGD_OUT=$V2_JUNE_BGD_OUT,V2_JUNE_BGD_BOUNDARY=$V2_JUNE_BGD_BOUNDARY" \
    slurm/cpcv2_june2023_bangladesh_eval.sbatch)"
job_id="${result%%;*}"

echo "submitted: $job_id"
echo "output:    $V2_JUNE_BGD_OUT"
echo "monitor:   squeue -j $job_id"
echo "log:       logs/bdhires-v2-jun23-bgd-$job_id.out"
