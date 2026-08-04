#!/usr/bin/env bash
#SBATCH --job-name=bdhires-2017-summary
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"
PYTHON_BIN="$ENV_PREFIX/bin/python"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
unset PYTHONHOME
unset PYTHONPATH

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bdhires-mpl-${SLURM_JOB_ID}"
ROOT="${BMD_2017_SMOKE_ROOT:-data/processed/bmd_imerg_smoke_2017_may_sep}"
CARTOPY_DIR="${CARTOPY_DATA_DIR:-data/static/cartopy}"
mkdir -p "$MPLCONFIGDIR" "$ROOT" "$CARTOPY_DIR" logs

EVALUATIONS=()
DUMPS=()
REPORTS=()
for fold in 0 1 2 3 4; do
    prefix="$ROOT/fold${fold}"
    for required in "${prefix}_evaluation.json" "${prefix}.npz" "${prefix}.json"; do
        [[ -s "$required" ]] || { echo "ERROR: missing $required"; exit 1; }
    done
    EVALUATIONS+=("${prefix}_evaluation.json")
    DUMPS+=("${prefix}.npz")
    REPORTS+=("${prefix}.json")
done

srun --ntasks=1 "$PYTHON_BIN" -u scripts/20_summarize_bmd_rotated_folds.py \
    --evaluations "${EVALUATIONS[@]}" \
    --out-json "$ROOT/rotated_summary.json" \
    --out-plot "$ROOT/rotated_summary.png"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/21_bmd_fullmonth_diagnostics.py \
    --dumps "${DUMPS[@]}" \
    --reports "${REPORTS[@]}" \
    --summary "$ROOT/rotated_summary.json" \
    --out-verification "$ROOT/verification.png" \
    --out-spatial "$ROOT/spatial_impact.png" \
    --out-report "$ROOT/diagnostics.json" \
    --cartopy-data-dir "$CARTOPY_DIR"

echo "2017 May-September process-test summary complete."
echo "Primary summary: $ROOT/rotated_summary.json"
echo "Verification suite: $ROOT/verification.png"
echo "Spatial impact: $ROOT/spatial_impact.png"
