#!/bin/bash
#SBATCH --job-name=bdhires-osse-footprint-finalize
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"
PYTHON_BIN="$ENV_PREFIX/bin/python"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"

ROOT="${OSSE_FOOTPRINT_ROOT:-data/processed/osse_footprint_ablation_50_2021_2024}"
LABELS=(
    gauges_exact_50_common
    satellite_exact_50_f2 satellite_exact_50_f4 satellite_exact_50_f10
    simultaneous_exact_50_f2 simultaneous_exact_50_f4 simultaneous_exact_50_f10
)

for LABEL in "${LABELS[@]}"; do
    OUT="$ROOT/$LABEL"
    "$PYTHON_BIN" -u scripts/25_merge_osse_chunks.py \
        --chunks-root "$ROOT/chunks" --arm "$LABEL" --out-dir "$OUT"
    "$PYTHON_BIN" -u scripts/14_osse_summary.py \
        --dump "$OUT/ensemble.npz" --cases 6 \
        --out-reconstruction "$OUT/reconstruction.png" \
        --out-matrix "$OUT/scale_matrix.png" \
        --out-report "$OUT/scale_summary.json"
    "$PYTHON_BIN" -u scripts/22_evaluate_osse_downscaling.py \
        --dump "$OUT/ensemble.npz" --label "$LABEL" \
        --minimum-valid-fraction 1.0 \
        --out-figure "$OUT/downscaling_diagnostics.png" \
        --out-report "$OUT/downscaling.json" \
        --out-curve-data "$OUT/downscaling_curves.npz" \
        --out-spatial-data "$OUT/spatial_fields.nc"
done

"$PYTHON_BIN" -u scripts/26_osse_footprint_ablation.py \
    --root "$ROOT" --out-dir "$ROOT/paper"

echo "Footprint ablation complete: $ROOT/paper"
find "$ROOT/paper" -maxdepth 1 -type f -print | sort
