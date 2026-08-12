#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v1-v2-compare
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
unset PYTHONHOME
unset PYTHONPATH

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bdhires-v1-v2-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" logs

V1_RAW_ROOT="${V1_RAW_ROOT:-data/processed/bmd_imerg_eval_ing2022_RAW}"
V1_S04_ROOT="${V1_S04_ROOT:-data/processed/bmd_imerg_eval_ing2022_S04}"
V2_ROOT="${V2_INGEST_ROOT:-data/processed/v2_ingestion_triplet/ing2022_s04_g010_sqrtfix}"
OUT_ROOT="${V1_V2_COMPARE_ROOT:-$V2_ROOT/v1_vs_v2}"

V1_RAW_DUMPS=()
V1_S04_DUMPS=()
V2_DUMPS=()
for fold in 0 1 2 3 4; do
    v1_raw="$V1_RAW_ROOT/fold${fold}.npz"
    v1_s04="$V1_S04_ROOT/fold${fold}.npz"
    v2="$V2_ROOT/fold${fold}.npz"
    for required in "$v1_raw" "$v1_s04" "$v2"; do
        [[ -s "$required" ]] || { echo "ERROR: missing $required"; exit 1; }
    done
    V1_RAW_DUMPS+=("$v1_raw")
    V1_S04_DUMPS+=("$v1_s04")
    V2_DUMPS+=("$v2")
done
mkdir -p "$OUT_ROOT"

"$PYTHON_BIN" -u scripts/52_compare_v1_v2_da.py \
    --v1-raw-dumps "${V1_RAW_DUMPS[@]}" \
    --v1-s04-dumps "${V1_S04_DUMPS[@]}" \
    --v2-dumps "${V2_DUMPS[@]}" \
    --block-days "${V1_V2_BLOCK_DAYS:-3}" \
    --n-resamples "${V1_V2_BOOTSTRAPS:-10000}" \
    --out-json "$OUT_ROOT/v1_vs_v2_selection.json" \
    --out-markdown "$OUT_ROOT/v1_vs_v2_selection.md" \
    --out-plot "$OUT_ROOT/v1_vs_v2_selection.png"

echo "Matched v1-v2 comparison complete."
echo "  table:  $OUT_ROOT/v1_vs_v2_selection.md"
echo "  data:   $OUT_ROOT/v1_vs_v2_selection.json"
echo "  figure: $OUT_ROOT/v1_vs_v2_selection.png"
