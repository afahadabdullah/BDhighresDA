#!/bin/bash
#SBATCH --job-name=bdhires-osse-summary
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"
PYTHON_BIN="$ENV_PREFIX/bin/python"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"

ROOT="${OSSE_PAPER_ROOT:-data/processed/osse_paper}"
PRIMARY="${OSSE_PAPER_PRIMARY:-simultaneous_realistic_40}"

# Older or interrupted finalization jobs may leave only one paper-ready arm.
# Resume the CPU-only merge/evaluation from existing monthly GPU chunks.  The
# presence checks make this cheap when an arm is already complete.
if [[ -d "$ROOT/chunks/gauges_exact_bmd" || -d "$ROOT/gauges_exact_bmd" ]]; then
    SOURCE_ARMS=(gauges_exact_bmd satellite_exact_bmd simultaneous_exact_bmd)
else
    SOURCE_ARMS=(gauges_realistic_40 satellite_realistic_40 simultaneous_realistic_40)
fi

for ARM in "${SOURCE_ARMS[@]}"; do
    OUT="$ROOT/$ARM"
    if [[ ! -s "$OUT/ensemble.npz" ]]; then
        if compgen -G "$ROOT/chunks/$ARM/*/ensemble.npz" >/dev/null; then
            echo "merging existing GPU chunks: $ARM"
            srun --ntasks=1 "$PYTHON_BIN" -u scripts/25_merge_osse_chunks.py \
                --chunks-root "$ROOT/chunks" \
                --arm "$ARM" \
                --out-dir "$OUT"
        else
            echo "ERROR: no merged ensemble or GPU chunks for $ARM" >&2
            exit 2
        fi
    fi
    if [[ ! -s "$OUT/scale_summary.json" && ! -s "$OUT/osse_scale_summary.json" ]]; then
        echo "rebuilding missing scale diagnostics: $ARM"
        srun --ntasks=1 "$PYTHON_BIN" -u scripts/14_osse_summary.py \
            --dump "$OUT/ensemble.npz" \
            --cases 6 \
            --out-reconstruction "$OUT/reconstruction.png" \
            --out-matrix "$OUT/scale_matrix.png" \
            --out-report "$OUT/scale_summary.json"
    fi
    if [[ ! -s "$OUT/downscaling.json" ]]; then
        echo "rebuilding missing downscaling diagnostics: $ARM"
        srun --ntasks=1 "$PYTHON_BIN" -u scripts/22_evaluate_osse_downscaling.py \
            --dump "$OUT/ensemble.npz" \
            --label "$ARM" \
            --minimum-valid-fraction 1.0 \
            --out-figure "$OUT/downscaling_diagnostics.png" \
            --out-report "$OUT/downscaling.json" \
            --out-curve-data "$OUT/downscaling_curves.npz" \
            --out-spatial-data "$OUT/spatial_fields.nc"
    fi
done

# One driver assembles every arm into the paper evidence set: ablation and
# downscaling/source-value tables (LaTeX), figure drafts, tidy CSVs, combined
# curve NPZ, and RESULTS.md.  Ensemble scores are not recomputed here; source
# attribution is derived from each arm's existing scale_summary.json.
srun --ntasks=1 "$PYTHON_BIN" -u scripts/24_osse_paper_suite.py \
    --root "$ROOT" \
    --primary "$PRIMARY" \
    --out-dir "$ROOT/paper"

echo
echo "Paper artifacts in $ROOT/paper:"
ls -1 "$ROOT/paper"
