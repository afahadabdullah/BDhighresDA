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
export V2_INGEST_ROOT="${V2_INGEST_ROOT:-data/processed/v2_ingestion_triplet/ing2022_s04}"
export BMD_CKPT="${BMD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
export BMD_CONFIG="${BMD_CONFIG:-configs/da.yaml}"
export BMD_DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
export BMD_STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
export BACKGROUND_DAY_OFFSET="${BACKGROUND_DAY_OFFSET:--1}"

PREP_PYTHON="${V2_INGEST_PREP_PYTHON:-/home/afahad/nb/project/BDDA/envs/bdda-gh200/bin/python}"
[[ -x "$PREP_PYTHON" ]] || PREP_PYTHON="python"

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

if [[ ! -s "$NATIVE" ]]; then
    PYTHONPATH="$PWD/src" "$PREP_PYTHON" -u scripts/43_subset_prepared_imerg.py \
        --input "$ARCHIVE_GLOB" \
        --start "$V2_INGEST_START" --end "$V2_INGEST_END" \
        --out "$NATIVE" --report "${NATIVE%.nc}_qc.json"
fi
if [[ ! -s "$V2_INGEST_IMERG" ]]; then
    PYTHONPATH="$PWD/src" "$PREP_PYTHON" -u scripts/44_coarsen_imerg_observations.py \
        --input "$NATIVE" --factor 8 --out "$V2_INGEST_IMERG"
fi

echo "Submitting CPC-v2 BMD/IMERG ingestion triplet"
echo "  period: $V2_INGEST_START through $V2_INGEST_END"
echo "  members: $V2_INGEST_MEMBERS; checkpoint: $BMD_CKPT"
echo "  IMERG: S04 0.4-degree file $V2_INGEST_IMERG"
echo "  outputs: $V2_INGEST_ROOT"

ARRAY_OVERRIDE=()
if [[ "${V2_INGEST_PREFLIGHT:-0}" == "1" ]]; then
    ARRAY_OVERRIDE=(--array=0)
    echo "  mode: numerical preflight (fold 0 only; no pooled summary)"
fi

array_result="$(sbatch --parsable --export=ALL "$@" \
    "${ARRAY_OVERRIDE[@]+"${ARRAY_OVERRIDE[@]}"}" \
    slurm/v2_ingestion_triplet.sbatch)"
array_job="${array_result%%;*}"
if [[ "${V2_INGEST_PREFLIGHT:-0}" == "1" ]]; then
    echo "submitted fold-0 preflight: $array_job"
    echo "inspect: logs/bdhires-v2-ingest-${array_job}_0.out"
    exit 0
fi

summary_result="$(sbatch --parsable --dependency="afterok:${array_job}" --export=ALL \
    "$@" slurm/v2_ingestion_triplet_summary.sbatch)"
summary_job="${summary_result%%;*}"

echo "submitted five-fold GPU array: $array_job"
echo "submitted dependent pooled summary: $summary_job"
echo "final comparison: $V2_INGEST_ROOT/ingestion_selection.{md,json,png}"
