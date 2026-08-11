#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v2-ingest-summary
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bdhires-v2-ingest-summary-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" logs

ROOT="${V2_INGEST_ROOT:-data/processed/v2_ingestion_triplet/ing2022_s04_g010_capped}"
DUMPS=()
REPORTS=()
for fold in 0 1 2 3 4; do
    dump="$ROOT/fold${fold}.npz"
    report="$ROOT/fold${fold}.json"
    [[ -s "$dump" ]] || { echo "ERROR: missing $dump"; exit 1; }
    [[ -s "$report" ]] || { echo "ERROR: missing $report"; exit 1; }
    DUMPS+=("$dump")
    REPORTS+=("$report")
done

"$PYTHON_BIN" -u scripts/50_summarize_v2_ingestion_triplet.py \
    --dumps "${DUMPS[@]}" --reports "${REPORTS[@]}" \
    --block-days "${V2_INGEST_BLOCK_DAYS:-3}" \
    --n-resamples "${V2_INGEST_BOOTSTRAPS:-10000}" \
    --out-json "$ROOT/ingestion_selection.json" \
    --out-markdown "$ROOT/ingestion_selection.md" \
    --out-plot "$ROOT/ingestion_selection.png" \
    --fold-plot-dir "$ROOT/fold_plots"

echo "CPC-v2 ingestion selection complete."
echo "  table:  $ROOT/ingestion_selection.md"
echo "  data:   $ROOT/ingestion_selection.json"
echo "  figure: $ROOT/ingestion_selection.png"
echo "  folds:  $ROOT/fold_plots/fold{0..4}_diagnostics.png"
