#!/usr/bin/env bash
# Submit the three monthly and one May--August BRISHTI-05 evaluations.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

export V2_CONFIRM_ROOT="${V2_CONFIRM_ROOT:-data/processed/v2_confirmatory_2021_2024}"
CPC_DIR="${V2_JUNE_BGD_CPC_DIR:-data/raw/cpc}"
BOUNDARY="${BRISHTI_BOUNDARY:-$V2_CONFIRM_ROOT/evaluation/brishti05_inputs/geoBoundaries-BGD-ADM0.geojson}"
PREP_PYTHON="${V2_JUNE_BGD_PREP_PYTHON:-$(command -v python3 || command -v python)}"

[[ -n "$PREP_PYTHON" && -x "$PREP_PYTHON" ]] || {
    echo "ERROR: no login-node Python found; set V2_JUNE_BGD_PREP_PYTHON" >&2
    exit 1
}
if [[ ! -s "$BOUNDARY" ]]; then
    "$PREP_PYTHON" scripts/80_fetch_bangladesh_boundary.py --output "$BOUNDARY"
fi
for year in 2021 2022 2023 2024; do
    [[ -s "$CPC_DIR/precip.${year}.nc" ]] || {
        "$PREP_PYTHON" scripts/02b_download_cpc.py --start 2021 --end 2024 \
            --out "$CPC_DIR" --require-complete
        break
    }
done

for label in 2021_may_sep 2022_may_sep 2023_may_sep 2024_may_jun; do
    [[ -d "$V2_CONFIRM_ROOT/gridded/${label}.zarr" ]] || { echo "ERROR: missing $label Zarr" >&2; exit 1; }
    [[ -s "$V2_CONFIRM_ROOT/imerg_native/${label}.nc" ]] || { echo "ERROR: missing $label IMERG" >&2; exit 1; }
done

for spec in "may:5" "jul:7" "aug:8" "may_aug:5 6 7 8"; do
    tag="${spec%%:*}"
    months="${spec#*:}"
    out="$V2_CONFIRM_ROOT/evaluation/brishti05_${tag}2023_native_refs"
    result="$(sbatch --parsable \
        --export="ALL,V2_CONFIRM_ROOT=$V2_CONFIRM_ROOT,V2_JUNE_BGD_OUT=$out,V2_JUNE_BGD_BOUNDARY=$BOUNDARY,V2_JUNE_BGD_CPC_DIR=$CPC_DIR,BRISHTI_MONTHS=$months,BRISHTI_PERIOD_TAG=$tag" \
        slurm/cpcv2_june2023_bangladesh_eval.sbatch)"
    job_id="${result%%;*}"
    echo "$tag: submitted $job_id -> $out"
done
