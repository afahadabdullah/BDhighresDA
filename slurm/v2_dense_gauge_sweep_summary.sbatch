#!/usr/bin/env bash
#SBATCH --job-name=bdhires-dense-summary
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
unset PYTHONHOME PYTHONPATH
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"

ROOT="${V2_DENSE_ROOT:-data/processed/v2_dense_gauge_sweep}"
STATION_ROOT="$ROOT/stations"
LABELS=(base repr_measured repr_half clip2000 superob ensrf)

PROFILE_ARGS=()
for label in "${LABELS[@]}"; do
    prefix="$ROOT/profiles/$label"
    if [[ -s "${prefix}.npz" && -s "${prefix}.json" ]]; then
        PROFILE_ARGS+=(--profile "${label}=${prefix}")
    else
        echo "WARNING: skipping incomplete profile $label" >&2
    fi
done
[[ ${#PROFILE_ARGS[@]} -gt 0 ]] || { echo "ERROR: no completed profiles"; exit 1; }

"$PYTHON_BIN" -u scripts/88_summarize_dense_gauge_sweep.py \
    "${PROFILE_ARGS[@]}" \
    --station-summary "$STATION_ROOT/station_summary.csv" \
    --manifest "$STATION_ROOT/preparation_manifest.json" \
    --reference-profile base --reference-arm prod_huber3 \
    --out-dir "$ROOT/summary"

echo "Summary: $ROOT/summary/dense_gauge_sweep.{md,json,csv,png}"
