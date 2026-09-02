#!/usr/bin/env bash
# Submit Bangladesh ADM0 evaluation for BMD+BWDB 2021-2024 data across monthly and multi-month periods.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

export V2_BWDB_ROOT="${V2_BWDB_ROOT:-data/processed/v2_bmd_bwdb_huber3_2021_2024}"
export V2_IMERG_NATIVE_ROOT="${V2_IMERG_NATIVE_ROOT:-data/processed/v2_confirmatory_2021_2024/imerg_native}"
CPC_DIR="${V2_BWDB_CPC_DIR:-data/raw/cpc}"
BOUNDARY="${BRISHTI_BOUNDARY:-$V2_BWDB_ROOT/evaluation/brishti05_inputs/geoBoundaries-BGD-ADM0.geojson}"
PREP_PYTHON="${V2_BWDB_PREP_PYTHON:-$(command -v python3 || command -v python)}"
TARGET_YEAR="${BRISHTI_YEAR:-2023}"

[[ -n "$PREP_PYTHON" && -x "$PREP_PYTHON" ]] || {
    echo "ERROR: no login-node Python found; set V2_BWDB_PREP_PYTHON" >&2
    exit 1
}
if [[ ! -s "$BOUNDARY" ]]; then
    mkdir -p "$(dirname "$BOUNDARY")"
    if [[ -s "data/processed/v2_confirmatory_2021_2024/evaluation/brishti05_inputs/geoBoundaries-BGD-ADM0.geojson" ]]; then
        cp "data/processed/v2_confirmatory_2021_2024/evaluation/brishti05_inputs/geoBoundaries-BGD-ADM0.geojson"* "$(dirname "$BOUNDARY")/"
    else
        "$PREP_PYTHON" scripts/80_fetch_bangladesh_boundary.py --output "$BOUNDARY"
    fi
fi
for year in 2021 2022 2023 2024; do
    [[ -s "$CPC_DIR/precip.${year}.nc" ]] || {
        "$PREP_PYTHON" scripts/02b_download_cpc.py --start 2021 --end 2024 \
            --out "$CPC_DIR" --require-complete
        break
    }
done

for label in 2021_may_sep 2022_may_sep 2023_may_sep 2024_may_jun; do
    [[ -d "$V2_BWDB_ROOT/gridded/${label}.zarr" ]] || { echo "ERROR: missing $label Zarr in $V2_BWDB_ROOT/gridded" >&2; exit 1; }
    [[ -s "$V2_IMERG_NATIVE_ROOT/${label}.nc" ]] || { echo "ERROR: missing $label IMERG in $V2_IMERG_NATIVE_ROOT" >&2; exit 1; }
done

# Default to the monthly and multi-month periods matching the contract evaluation
if [[ $# -eq 0 ]]; then
    PERIOD_SPECS=("may:5" "jun:6" "jul:7" "aug:8" "may_aug:5 6 7 8")
else
    PERIOD_SPECS=("$@")
fi

for spec in "${PERIOD_SPECS[@]}"; do
    tag="${spec%%:*}"
    months="${spec#*:}"
    out="$V2_BWDB_ROOT/evaluation/brishti05_${tag}${TARGET_YEAR}_native_refs"
    result="$(sbatch --parsable \
        --export="ALL,V2_BWDB_ROOT=$V2_BWDB_ROOT,V2_IMERG_NATIVE_ROOT=$V2_IMERG_NATIVE_ROOT,V2_BWDB_BGD_OUT=$out,V2_BWDB_BGD_BOUNDARY=$BOUNDARY,V2_BWDB_CPC_DIR=$CPC_DIR,BRISHTI_MONTHS=$months,BRISHTI_PERIOD_TAG=$tag,BRISHTI_YEAR=$TARGET_YEAR" \
        slurm/bmd_bwdb_bangladesh_eval.sbatch)"
    job_id="${result%%;*}"
    echo "$tag: submitted $job_id -> $out"
done
