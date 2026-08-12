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
for store in "${STORES[@]}"; do
    label="$(basename "$store" .zarr)"
    out="$V2_CONFIRM_ROOT/evaluation/$label"
    result="$(sbatch --parsable \
        --export="ALL,V2_EVAL_ZARRS=$store,V2_EVAL_OUT=$out" \
        slurm/v2_gridded_evaluation.sbatch)"
    echo "  $label: ${result%%;*} -> $out"
done

if [[ "$POOL" == "1" && ${#STORES[@]} -gt 1 ]]; then
    joined="$(IFS=:; echo "${STORES[*]}")"
    label="pooled_${#STORES[@]}_seasons"
    out="$V2_CONFIRM_ROOT/evaluation/$label"
    result="$(sbatch --parsable \
        --export="ALL,V2_EVAL_ZARRS=$joined,V2_EVAL_OUT=$out" \
        slurm/v2_gridded_evaluation.sbatch)"
    echo "  pooled: ${result%%;*} -> $out"
fi

echo "Monitor: squeue -u $USER"
echo "Logs:    logs/bdhires-v2-grid-eval-*.out"
