#!/usr/bin/env bash
#SBATCH --job-name=bdhires-bwdb-h3-summary
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge 2>/dev/null || true
export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
unset PYTHONHOME PYTHONPATH
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bdhires-bwdb-h3-summary-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

ROOT="${V2_BWDB_WINNER_ROOT:-data/processed/v2_bmd_bwdb_huber3_2021_2024}"
LABELS=(2021_may_sep 2022_may_sep 2023_may_sep 2024_may_jun)
DUMPS=() REPORTS=() MANIFESTS=() SUMMARIES=() STORES=()
for label in "${LABELS[@]}"; do
    DUMPS+=("$ROOT/evaluation/$label.npz")
    REPORTS+=("$ROOT/evaluation/$label.json")
    MANIFESTS+=("$ROOT/stations/$label/preparation_manifest.json")
    SUMMARIES+=("$ROOT/stations/$label/station_summary.csv")
    STORES+=("$ROOT/gridded/$label.zarr")
done
"$PYTHON_BIN" -u scripts/84_summarize_bmd_bwdb_huber3_2021_2024.py \
    --dumps "${DUMPS[@]}" --reports "${REPORTS[@]}" \
    --manifests "${MANIFESTS[@]}" --station-summaries "${SUMMARIES[@]}" \
    --zarr-stores "${STORES[@]}" --out-dir "$ROOT/summary"
echo "Summary: $ROOT/summary/huber3_2021_2024_scores.{md,json,png}"
echo "Catalog: $ROOT/summary/gridded_catalog.json"
echo "Folds:   $ROOT/summary/fold_plots/{period}_fold0_diagnostics.png"
