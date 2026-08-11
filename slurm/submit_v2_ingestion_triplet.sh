#!/usr/bin/env bash
set -euo pipefail

# Prepare the validated S04 observation file once, then submit five matched
# CPC-v2 folds and a dependent day-block-bootstrap summary.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

export V2_INGEST_START="${V2_INGEST_START:-2022-05-01}"
export V2_INGEST_END="${V2_INGEST_END:-2022-05-10}"
export V2_INGEST_MEMBERS="${V2_INGEST_MEMBERS:-30}"
export V2_INGEST_SEED="${V2_INGEST_SEED:-201805}"
export V2_INGEST_ROOT="${V2_INGEST_ROOT:-data/processed/v2_ingestion_triplet/ing2022_s04_g010_capped}"
export BMD_CKPT="${BMD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
export BMD_CONFIG="${BMD_CONFIG:-configs/da.yaml}"
export BMD_DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
export BMD_STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
export BACKGROUND_DAY_OFFSET="${BACKGROUND_DAY_OFFSET:--1}"

# Preparation runs on the node launching this script, not on the Grace-Hopper
# nodes that execute the GPU array.  A Python binary from the GH200 environment
# is therefore not portable to an x86 login node.  Use the active environment
# unless the caller explicitly selects another compatible interpreter.
if [[ -n "${V2_INGEST_PREP_PYTHON:-}" ]]; then
    PREP_PYTHON="$V2_INGEST_PREP_PYTHON"
elif command -v python >/dev/null 2>&1; then
    PREP_PYTHON="python"
else
    PREP_PYTHON="python3"
fi
PREP_PYTHON_CHECKED=0

SHARED_DIR="${V2_INGEST_IMERG_DIR:-data/processed/imerg_prepared_ing2022}"
ARCHIVE_GLOB="${IMERG_ARCHIVE_GLOB:-data/processed/imerg_bd_aligned_*.nc}"
START_CLEAN="${V2_INGEST_START//-/}"
END_CLEAN="${V2_INGEST_END//-/}"
NATIVE="$SHARED_DIR/imerg_aligned_${START_CLEAN}_${END_CLEAN}.nc"
export V2_INGEST_IMERG="$SHARED_DIR/imerg_0p4deg_${START_CLEAN}_${END_CLEAN}.nc"
mkdir -p "$SHARED_DIR" "$V2_INGEST_ROOT"

for required in "$BMD_CKPT" "$BMD_CONFIG" "$BMD_STATIONS"; do
    [[ -f "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done
[[ -d "$BMD_DATA_DIR" ]] || { echo "ERROR: missing $BMD_DATA_DIR"; exit 1; }

prepare_imerg_window() {
    local window_start="$1"
    local window_end="$2"
    local native_out="$3"
    local coarse_out="$4"

    if [[ ! -s "$native_out" || ! -s "$coarse_out" ]] && (( ! PREP_PYTHON_CHECKED )); then
        if ! command -v "$PREP_PYTHON" >/dev/null 2>&1; then
            echo "ERROR: preparation Python not found: $PREP_PYTHON"
            echo "Activate a Python environment with numpy and xarray, or set V2_INGEST_PREP_PYTHON."
            exit 1
        fi
        if ! "$PREP_PYTHON" -c 'import numpy, xarray' >/dev/null 2>&1; then
            echo "ERROR: preparation Python cannot run with numpy and xarray: $PREP_PYTHON"
            echo "Activate a compatible environment, or set V2_INGEST_PREP_PYTHON."
            exit 1
        fi
        PREP_PYTHON_CHECKED=1
    fi

    if [[ ! -s "$native_out" ]]; then
        PYTHONPATH="$PWD/src" "$PREP_PYTHON" -u scripts/43_subset_prepared_imerg.py \
            --input "$ARCHIVE_GLOB" \
            --start "$window_start" --end "$window_end" \
            --out "$native_out" --report "${native_out%.nc}_qc.json"
    fi
    if [[ ! -s "$coarse_out" ]]; then
        PYTHONPATH="$PWD/src" "$PREP_PYTHON" -u scripts/44_coarsen_imerg_observations.py \
            --input "$native_out" --factor 8 --out "$coarse_out"
    fi
}

prepare_imerg_window \
    "$V2_INGEST_START" "$V2_INGEST_END" "$NATIVE" "$V2_INGEST_IMERG"

echo "Submitting CPC-v2 BMD/IMERG ingestion triplet"
echo "  period: $V2_INGEST_START through $V2_INGEST_END"
echo "  members: $V2_INGEST_MEMBERS; checkpoint: $BMD_CKPT"
echo "  IMERG: S04 0.4-degree file $V2_INGEST_IMERG"
echo "  outputs: $V2_INGEST_ROOT"

if [[ "${V2_INGEST_PREFLIGHT:-0}" == "1" ]]; then
    echo "  mode: numerical preflight (fold 0 only; no pooled summary)"
    array_result="$(sbatch --parsable --export=ALL "$@" --array=0 \
        slurm/v2_ingestion_triplet.sbatch)"
    array_job="${array_result%%;*}"
    echo "submitted fold-0 preflight: $array_job"
    echo "inspect: logs/bdhires-v2-ingest-${array_job}_0.out"
    exit 0
fi

# A full request automatically gates the expensive five-fold array behind the
# exact fold-0/member reproduction through May 2. The second historical
# failure appeared only on day 2 and member 17, so a tiny smoke test would give
# false reassurance. The full array remains pending without consuming GPUs if
# this numerical preflight fails.
DEPENDENCY_ARGS=()
preflight_job=""
if [[ "${V2_INGEST_AUTO_PREFLIGHT:-1}" == "1" ]]; then
    PREFLIGHT_ROOT="$V2_INGEST_ROOT/preflight"
    PREFLIGHT_END="${V2_INGEST_PREFLIGHT_END:-2022-05-02}"
    PREFLIGHT_MEMBERS="${V2_INGEST_PREFLIGHT_MEMBERS:-30}"
    if [[ "$PREFLIGHT_END" < "$V2_INGEST_START" || "$PREFLIGHT_END" > "$V2_INGEST_END" ]]; then
        echo "ERROR: preflight end $PREFLIGHT_END is outside the full run window"
        exit 1
    fi
    PREFLIGHT_END_CLEAN="${PREFLIGHT_END//-/}"
    PREFLIGHT_NATIVE="$SHARED_DIR/imerg_aligned_${START_CLEAN}_${PREFLIGHT_END_CLEAN}.nc"
    PREFLIGHT_IMERG="$SHARED_DIR/imerg_0p4deg_${START_CLEAN}_${PREFLIGHT_END_CLEAN}.nc"
    prepare_imerg_window \
        "$V2_INGEST_START" "$PREFLIGHT_END" \
        "$PREFLIGHT_NATIVE" "$PREFLIGHT_IMERG"
    PREFLIGHT_EXPORT="ALL,V2_INGEST_END=$PREFLIGHT_END"
    PREFLIGHT_EXPORT+=",V2_INGEST_MEMBERS=$PREFLIGHT_MEMBERS"
    PREFLIGHT_EXPORT+=",V2_INGEST_ROOT=$PREFLIGHT_ROOT"
    PREFLIGHT_EXPORT+=",V2_INGEST_IMERG=$PREFLIGHT_IMERG"
    preflight_result="$(sbatch --parsable "$@" --array=0 \
        --export="$PREFLIGHT_EXPORT" \
        slurm/v2_ingestion_triplet.sbatch)"
    preflight_job="${preflight_result%%;*}"
    DEPENDENCY_ARGS=(--dependency="afterok:${preflight_job}")
fi

array_result="$(sbatch --parsable --export=ALL "$@" \
    "${DEPENDENCY_ARGS[@]+"${DEPENDENCY_ARGS[@]}"}" \
    slurm/v2_ingestion_triplet.sbatch)"
array_job="${array_result%%;*}"

summary_result="$(sbatch --parsable --dependency="afterok:${array_job}" --export=ALL \
    "$@" slurm/v2_ingestion_triplet_summary.sbatch)"
summary_job="${summary_result%%;*}"

if [[ -n "$preflight_job" ]]; then
    echo "submitted automatic fold-0 preflight: $preflight_job"
    echo "preflight IMERG: $PREFLIGHT_IMERG"
fi
echo "submitted five-fold GPU array: $array_job"
echo "submitted dependent pooled summary: $summary_job"
echo "final comparison: $V2_INGEST_ROOT/ingestion_selection.{md,json,png}"
