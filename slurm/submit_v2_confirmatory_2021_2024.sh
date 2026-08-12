#!/usr/bin/env bash
set -euo pipefail

# Prepare exact 0.4-degree IMERG files, submit the 20 CV and four production
# jobs, then summarize only after every array task completes successfully.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

export V2_CONFIRM_ROOT="${V2_CONFIRM_ROOT:-data/processed/v2_confirmatory_2021_2024}"
export V2_CONFIRM_MEMBERS="${V2_CONFIRM_MEMBERS:-30}"
export V2_CONFIRM_SEED="${V2_CONFIRM_SEED:-201805}"
export BMD_CKPT="${BMD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
export BMD_CONFIG="${BMD_CONFIG:-configs/da.yaml}"
export BMD_DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
export BMD_STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
export BACKGROUND_DAY_OFFSET="${BACKGROUND_DAY_OFFSET:--1}"

export V2_CONFIRM_NATIVE_2021="${V2_CONFIRM_NATIVE_2021:-data/processed/bmd_imerg_eval_2021_may_sep/imerg_aligned_20210501_20210930.nc}"
export V2_CONFIRM_NATIVE_2022="${V2_CONFIRM_NATIVE_2022:-data/processed/bmd_imerg_eval_2022_may_sep/imerg_aligned_20220501_20220930.nc}"
export V2_CONFIRM_NATIVE_2023="${V2_CONFIRM_NATIVE_2023:-data/processed/bmd_imerg_eval_2023_may_sep/imerg_aligned_20230501_20230930.nc}"
export V2_CONFIRM_NATIVE_2024="${V2_CONFIRM_NATIVE_2024:-data/processed/bmd_imerg_eval_2024_may_jun/imerg_aligned_20240501_20240630.nc}"

if [[ -n "${V2_CONFIRM_PREP_PYTHON:-}" ]]; then
    PREP_PYTHON="$V2_CONFIRM_PREP_PYTHON"
elif command -v python >/dev/null 2>&1; then
    PREP_PYTHON="python"
else
    PREP_PYTHON="python3"
fi
if ! command -v "$PREP_PYTHON" >/dev/null 2>&1 \
   || ! "$PREP_PYTHON" -c 'import numpy, xarray' >/dev/null 2>&1; then
    echo "ERROR: preparation Python needs numpy and xarray: $PREP_PYTHON" >&2
    echo "Activate a compatible login-node environment or set V2_CONFIRM_PREP_PYTHON." >&2
    exit 1
fi

for required in "$BMD_CKPT" "$BMD_CONFIG" "$BMD_STATIONS"; do
    [[ -f "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done
[[ -d "$BMD_DATA_DIR" ]] || { echo "ERROR: missing $BMD_DATA_DIR"; exit 1; }

LABELS=(2021_may_sep 2022_may_sep 2023_may_sep 2024_may_jun)
STARTS=(2021-05-01 2022-05-01 2023-05-01 2024-05-01)
ENDS=(2021-09-30 2022-09-30 2023-09-30 2024-06-30)
NATIVE_FILES=(
    "$V2_CONFIRM_NATIVE_2021"
    "$V2_CONFIRM_NATIVE_2022"
    "$V2_CONFIRM_NATIVE_2023"
    "$V2_CONFIRM_NATIVE_2024"
)
mkdir -p "$V2_CONFIRM_ROOT/imerg_native" "$V2_CONFIRM_ROOT/imerg_s04"
COARSE_FILES=()
for index in 0 1 2 3; do
    source_native="${NATIVE_FILES[$index]}"
    native="$V2_CONFIRM_ROOT/imerg_native/${LABELS[$index]}.nc"
    coarse="$V2_CONFIRM_ROOT/imerg_s04/${LABELS[$index]}.nc"
    [[ -s "$source_native" ]] || {
        echo "ERROR: prepared native IMERG source missing: $source_native" >&2
        echo "Set V2_CONFIRM_NATIVE_20XX to the matching BMD-aligned seasonal file." >&2
        exit 1
    }
    # This is deliberately not a filename-only check. Script 43 validates the
    # BMD 03:00-03:00 accumulation, units, variables, and every exact date,
    # preventing a queued seasonal array from discovering a bad time axis.
    if [[ ! -s "$native" ]]; then
        PYTHONPATH="$PWD/src" "$PREP_PYTHON" -u \
            scripts/43_subset_prepared_imerg.py \
            --input "$source_native" \
            --start "${STARTS[$index]}" --end "${ENDS[$index]}" \
            --out "$native" --report "${native%.nc}_qc.json"
    else
        echo "Reusing $native"
    fi
    if [[ ! -s "$coarse" ]]; then
        PYTHONPATH="$PWD/src" "$PREP_PYTHON" -u \
            scripts/44_coarsen_imerg_observations.py \
            --input "$native" --factor 8 --out "$coarse" \
            --report "${coarse%.nc}_qc.json"
    else
        echo "Reusing $coarse"
    fi
    COARSE_FILES+=("$coarse")
done

export V2_CONFIRM_IMERG_2021="${COARSE_FILES[0]}"
export V2_CONFIRM_IMERG_2022="${COARSE_FILES[1]}"
export V2_CONFIRM_IMERG_2023="${COARSE_FILES[2]}"
export V2_CONFIRM_IMERG_2024="${COARSE_FILES[3]}"

echo "Submitting frozen CPC-v2 confirmation archive"
echo "  periods: May-Sep 2021, 2022, 2023; May-Jun 2024"
echo "  array: 20 held-out-fold tasks + 4 all-station Zarr tasks"
echo "  members: $V2_CONFIRM_MEMBERS; checkpoint: $BMD_CKPT"
echo "  root: $V2_CONFIRM_ROOT"
echo "  completed tasks are reused; partial outputs fail loudly"

array_result="$(sbatch --parsable --export=ALL "$@" \
    slurm/v2_confirmatory_2021_2024.sbatch)"
array_job="${array_result%%;*}"
summary_result="$(sbatch --parsable --dependency="afterok:${array_job}" \
    --export=ALL "$@" slurm/v2_confirmatory_2021_2024_summary.sbatch)"
summary_job="${summary_result%%;*}"

echo "submitted GPU array: $array_job"
echo "submitted dependent summary: $summary_job"
echo "monitor: squeue -u $USER"
echo "logs:    logs/bdhires-v2-confirm-${array_job}_*.out"
echo "Zarr:    $V2_CONFIRM_ROOT/gridded/{2021_may_sep,2022_may_sep,2023_may_sep,2024_may_jun}.zarr"
echo "summary: $V2_CONFIRM_ROOT/summary/confirmatory_selection.{md,json,png}"
echo "folds:   $V2_CONFIRM_ROOT/summary/fold_plots/{period}_fold{0..4}_diagnostics.png"
