#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v2-confirm-summary
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bdhires-v2-confirm-summary-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" logs

ROOT="${V2_CONFIRM_ROOT:-data/processed/v2_confirmatory_2021_2024}"
LABELS=(2021_may_sep 2022_may_sep 2023_may_sep 2024_may_jun)
DUMPS=()
REPORTS=()
STORES=()
for label in "${LABELS[@]}"; do
    for fold in 0 1 2 3 4; do
        dump="$ROOT/cv/$label/fold${fold}.npz"
        report="$ROOT/cv/$label/fold${fold}.json"
        [[ -s "$dump" ]] || { echo "ERROR: missing $dump"; exit 1; }
        [[ -s "$report" ]] || { echo "ERROR: missing $report"; exit 1; }
        DUMPS+=("$dump")
        REPORTS+=("$report")
    done
    store="$ROOT/gridded/${label}.zarr"
    [[ -d "$store" ]] || { echo "ERROR: missing $store"; exit 1; }
    STORES+=("$store")
done

"$PYTHON_BIN" -u scripts/54_summarize_v2_confirmatory.py \
    --dumps "${DUMPS[@]}" \
    --reports "${REPORTS[@]}" \
    --zarr-stores "${STORES[@]}" \
    --block-days "${V2_CONFIRM_BLOCK_DAYS:-3}" \
    --n-resamples "${V2_CONFIRM_BOOTSTRAPS:-10000}" \
    --out-dir "$ROOT/summary"

echo "Frozen CPC-v2 confirmation summary complete."
echo "  table:   $ROOT/summary/confirmatory_selection.md"
echo "  data:    $ROOT/summary/confirmatory_selection.json"
echo "  figure:  $ROOT/summary/confirmatory_selection.png"
echo "  catalog: $ROOT/summary/gridded_catalog.json"
echo "  folds:   $ROOT/summary/fold_plots/{period}_fold{0..4}_diagnostics.png"
