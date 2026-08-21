#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

export V2_CONFIRM_ROOT="${V2_CONFIRM_ROOT:-data/processed/v2_confirmatory_2021_2024}"
export V2_JUNE_BGD_OUT="${V2_JUNE_BGD_OUT:-$V2_CONFIRM_ROOT/evaluation/brishti05_june2023_native_refs}"
export V2_JUNE_BGD_BOUNDARY="${V2_JUNE_BGD_BOUNDARY:-$V2_JUNE_BGD_OUT/inputs/geoBoundaries-BGD-ADM0.geojson}"
export V2_JUNE_BGD_CPC_DIR="${V2_JUNE_BGD_CPC_DIR:-data/raw/cpc}"

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

missing_cpc=0
for year in 2021 2022 2023 2024; do
    [[ -s "$V2_JUNE_BGD_CPC_DIR/precip.${year}.nc" ]] || missing_cpc=1
done
if [[ "$missing_cpc" == "1" ]]; then
    echo "Downloading missing original NOAA CPC 0.5-degree annual files"
    "$PREP_PYTHON" scripts/02b_download_cpc.py \
        --start 2021 --end 2024 --out "$V2_JUNE_BGD_CPC_DIR" --require-complete
fi

for label in 2021_may_sep 2022_may_sep 2023_may_sep 2024_may_jun; do
    [[ -d "$V2_CONFIRM_ROOT/gridded/${label}.zarr" ]] || {
        echo "ERROR: missing saved analysis $V2_CONFIRM_ROOT/gridded/${label}.zarr" >&2
        exit 1
    }
    [[ -s "$V2_CONFIRM_ROOT/imerg_native/${label}.nc" ]] || {
        echo "ERROR: missing native IMERG $V2_CONFIRM_ROOT/imerg_native/${label}.nc" >&2
        exit 1
    }
done

result="$(sbatch --parsable \
    --export="ALL,V2_CONFIRM_ROOT=$V2_CONFIRM_ROOT,V2_JUNE_BGD_OUT=$V2_JUNE_BGD_OUT,V2_JUNE_BGD_BOUNDARY=$V2_JUNE_BGD_BOUNDARY,V2_JUNE_BGD_CPC_DIR=$V2_JUNE_BGD_CPC_DIR" \
    slurm/cpcv2_june2023_bangladesh_eval.sbatch)"
job_id="${result%%;*}"

echo "submitted: $job_id"
echo "output:    $V2_JUNE_BGD_OUT"
echo "monitor:   squeue -j $job_id"
echo "log:       logs/bdhires-brishti05-jun23-$job_id.out"
