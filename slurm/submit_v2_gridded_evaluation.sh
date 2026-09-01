#!/usr/bin/env bash
set -euo pipefail

# Submit one evaluation per currently completed seasonal Zarr, and optionally a
# pooled evaluation of every completed season. Safe to rerun as new seasons end.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

export V2_CONFIRM_ROOT="${V2_CONFIRM_ROOT:-data/processed/v2_confirmatory_2021_2024}"
export V2_EVAL_CV_ROOT="${V2_EVAL_CV_ROOT:-$V2_CONFIRM_ROOT}"
export V2_EVAL_TEXTURE_MEMBERS="${V2_EVAL_TEXTURE_MEMBERS:-5}"
POOL="${V2_EVAL_POOL:-1}"
PER_SEASON="${V2_EVAL_PER_SEASON:-1}"
COMPARISON_ROOT="${V2_EVAL_COMPARISON_ROOT:-}"

comparison_for_store() {
    local eval_store="$1"
    if [[ -z "$COMPARISON_ROOT" ]]; then
        printf '%s' "${V2_EVAL_COMPARISON_ZARR:-}"
        return
    fi
    local label
    label="$(basename "$eval_store" .zarr)"
    local comparison="$COMPARISON_ROOT/gridded/$label.zarr"
    [[ -d "$comparison" ]] || { echo "ERROR: matched comparison store missing: $comparison" >&2; exit 1; }
    printf '%s' "$comparison"
}

STORES=()
if (( $# )); then
    for store in "$@"; do
        [[ -d "$store" ]] || { echo "ERROR: not a completed Zarr directory: $store"; exit 1; }
        STORES+=("$store")
    done
else
    for store in "$V2_CONFIRM_ROOT"/gridded/*.zarr; do
        [[ -d "$store" ]] && STORES+=("$store")
    done
fi
if (( ${#STORES[@]} == 0 )); then
    echo "ERROR: no completed seasonal Zarr stores found under $V2_CONFIRM_ROOT/gridded" >&2
    exit 1
fi

echo "Submitting evaluation for ${#STORES[@]} completed season(s)"
if [[ "$PER_SEASON" == "1" ]]; then
    for store in "${STORES[@]}"; do
        label="$(basename "$store" .zarr)"
        out="$V2_CONFIRM_ROOT/evaluation/$label"
        comparison="$(comparison_for_store "$store")"
        result="$(sbatch --parsable \
            --export="ALL,V2_EVAL_ZARRS=$store,V2_EVAL_OUT=$out,V2_EVAL_COMPARISON_ZARR=$comparison" \
            slurm/v2_gridded_evaluation.sbatch)"
        echo "  $label: ${result%%;*} -> $out"
    done
else
    echo "  per-season jobs disabled (V2_EVAL_PER_SEASON=0)"
fi

if [[ "$POOL" == "1" && ${#STORES[@]} -gt 1 ]]; then
    joined="$(IFS=:; echo "${STORES[*]}")"
    comparisons=()
    for store in "${STORES[@]}"; do
        comparison="$(comparison_for_store "$store")"
        [[ -n "$comparison" ]] && comparisons+=("$comparison")
    done
    comparison_joined="$(IFS=:; echo "${comparisons[*]}")"
    label="pooled_${#STORES[@]}_seasons"
    out="$V2_CONFIRM_ROOT/evaluation/$label"
    result="$(sbatch --parsable \
        --export="ALL,V2_EVAL_ZARRS=$joined,V2_EVAL_OUT=$out,V2_EVAL_COMPARISON_ZARR=$comparison_joined" \
        slurm/v2_gridded_evaluation.sbatch)"
    echo "  pooled: ${result%%;*} -> $out"
fi

echo "Monitor: squeue -u $USER"
echo "Logs:    logs/bdhires-v2-grid-eval-*.out"
