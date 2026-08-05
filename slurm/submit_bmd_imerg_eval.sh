#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

BMD_START="${BMD_START:-2021-05-01}"
BMD_END="${BMD_END:-2021-09-30}"
BMD_EVAL_LABEL="${BMD_EVAL_LABEL:-2021_may_sep}"
BMD_DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
BMD_STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
BMD_MEMBERS="${BMD_MEMBERS:-16}"

ROOT="data/processed/bmd_imerg_eval_${BMD_EVAL_LABEL}"
mkdir -p "$ROOT"

# 1. Convert new station directory if daily CSV doesn't exist yet
DAILY_CSV="$ROOT/bmd_daily.csv"
STATION_CSV="$ROOT/bmd_stations.csv"
QC_JSON="$ROOT/bmd_qc.json"

export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"

if [[ -x "$PYTHON_BIN" ]]; then
    "$PYTHON_BIN" scripts/05_convert_bmd_dir.py \
        --data-dir "$BMD_DATA_DIR" \
        --stations "$BMD_STATIONS" \
        --start "$BMD_START" \
        --end "$BMD_END" \
        --out "$DAILY_CSV" \
        --summary "$STATION_CSV" \
        --report "$QC_JSON"
fi

array_result="$(sbatch --parsable \
    --export="ALL,BMD_START=$BMD_START,BMD_END=$BMD_END,BMD_EVAL_LABEL=$BMD_EVAL_LABEL,BMD_DATA_DIR=$BMD_DATA_DIR,BMD_STATIONS=$BMD_STATIONS,BMD_MEMBERS=$BMD_MEMBERS" \
    "$@" \
    slurm/bmd_imerg_rotated_folds_eval.sbatch)"
array_job="${array_result%%;*}"

summary_result="$(sbatch --parsable --dependency="afterok:${array_job}" \
    --export="ALL,BMD_EVAL_LABEL=$BMD_EVAL_LABEL" \
    "$@" \
    slurm/bmd_imerg_rotated_folds_eval_summary.sbatch)"
summary_job="${summary_result%%;*}"

echo "submitted five-fold GPU evaluation array: ${array_job}"
echo "submitted dependent CPU summary & diagnostics: ${summary_job}"
echo "period: ${BMD_START} through ${BMD_END}"
echo "ensemble members: ${BMD_MEMBERS}"
echo "outputs: ${ROOT}/"
