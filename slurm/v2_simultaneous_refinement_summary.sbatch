#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v2-simul-summary
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
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bdhires-v2-simul-summary-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" logs

ROOT="${V2_REFINE_ROOT:-data/processed/v2_simultaneous_refinement/ing2022_s04}"
REFERENCE_ROOT="${V2_REFINE_REFERENCE_ROOT:-data/processed/v2_ingestion_triplet/ing2022_s04_g010_sqrtfix}"
REFERENCE_DUMPS=()
REFERENCE_REPORTS=()
CANDIDATE_DUMPS=()
CANDIDATE_REPORTS=()
for fold in 0 1 2 3 4; do
    reference_dump="$REFERENCE_ROOT/fold${fold}.npz"
    reference_report="$REFERENCE_ROOT/fold${fold}.json"
    candidate_dump="$ROOT/fold${fold}.npz"
    candidate_report="$ROOT/fold${fold}.json"
    for required in \
        "$reference_dump" "$reference_report" "$candidate_dump" "$candidate_report"; do
        [[ -s "$required" ]] || { echo "ERROR: missing $required"; exit 1; }
    done
    REFERENCE_DUMPS+=("$reference_dump")
    REFERENCE_REPORTS+=("$reference_report")
    CANDIDATE_DUMPS+=("$candidate_dump")
    CANDIDATE_REPORTS+=("$candidate_report")
done

"$PYTHON_BIN" -u scripts/53_summarize_v2_simultaneous_refinement.py \
    --reference-dumps "${REFERENCE_DUMPS[@]}" \
    --reference-reports "${REFERENCE_REPORTS[@]}" \
    --candidate-dumps "${CANDIDATE_DUMPS[@]}" \
    --candidate-reports "${CANDIDATE_REPORTS[@]}" \
    --block-days "${V2_REFINE_BLOCK_DAYS:-3}" \
    --n-resamples "${V2_REFINE_BOOTSTRAPS:-10000}" \
    --out-json "$ROOT/refinement_selection.json" \
    --out-markdown "$ROOT/refinement_selection.md" \
    --out-plot "$ROOT/refinement_selection.png" \
    --fold-plot-dir "$ROOT/fold_plots"

echo "CPC-v2 simultaneous refinement selection complete."
echo "  table:  $ROOT/refinement_selection.md"
echo "  data:   $ROOT/refinement_selection.json"
echo "  figure: $ROOT/refinement_selection.png"
echo "  folds:  $ROOT/fold_plots/fold{0..4}_diagnostics.png"
