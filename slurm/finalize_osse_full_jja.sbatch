#!/bin/bash
#SBATCH --job-name=bdhires-osse-full-finalize
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"
PYTHON_BIN="$ENV_PREFIX/bin/python"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"

ROOT="${OSSE_FULL_ROOT:-data/processed/osse_full_jja_2021_2024}"
NETWORK_TAG="${OSSE_FULL_NETWORK_TAG:-bmd}"
PRIMARY="simultaneous_exact_${NETWORK_TAG}"

for LABEL in "gauges_exact_${NETWORK_TAG}" "satellite_exact_${NETWORK_TAG}" "$PRIMARY"; do
    OUT="$ROOT/$LABEL"
    "$PYTHON_BIN" -u scripts/25_merge_osse_chunks.py \
        --chunks-root "$ROOT/chunks" \
        --arm "$LABEL" \
        --out-dir "$OUT"

    "$PYTHON_BIN" -u scripts/11_da_diagnostics.py \
        --dump "$OUT/ensemble.npz" \
        --out-figure "$OUT/da_diagnostics.png" \
        --out-report "$OUT/da_diagnostics.json"

    "$PYTHON_BIN" -u scripts/14_osse_summary.py \
        --dump "$OUT/ensemble.npz" \
        --cases 6 \
        --out-reconstruction "$OUT/reconstruction.png" \
        --out-matrix "$OUT/scale_matrix.png" \
        --out-report "$OUT/scale_summary.json"

    "$PYTHON_BIN" -u scripts/22_evaluate_osse_downscaling.py \
        --dump "$OUT/ensemble.npz" \
        --label "$LABEL" \
        --minimum-valid-fraction 1.0 \
        --out-figure "$OUT/downscaling_diagnostics.png" \
        --out-report "$OUT/downscaling.json" \
        --out-curve-data "$OUT/downscaling_curves.npz" \
        --out-spatial-data "$OUT/spatial_fields.nc"

    if [[ "$LABEL" == "$PRIMARY" ]]; then
        "$PYTHON_BIN" -u scripts/12_da_impact_maps.py \
            --dump "$OUT/ensemble.npz" \
            --cases 6 \
            --out-cases "$OUT/impact_cases.png" \
            --out-aggregate "$OUT/impact_aggregate.png" \
            --out-report "$OUT/impact.json"
    fi
done

"$PYTHON_BIN" -u scripts/24_osse_paper_suite.py \
    --root "$ROOT" \
    --primary "$PRIMARY" \
    --out-dir "$ROOT/paper"

echo
echo "Maximum full-JJA OSSE complete: $ROOT"
find "$ROOT/paper" -maxdepth 1 -type f -print | sort
