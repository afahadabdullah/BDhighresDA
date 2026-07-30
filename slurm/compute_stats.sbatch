#!/usr/bin/env bash
#SBATCH --job-name=compute-stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

[[ -f scripts/06_compute_stats.py ]] || {
    echo "ERROR: Submit this job from the BDhighresDA repository root."
    exit 1
}

ZARR_STORE="${STATS_ZARR:-data/processed/bd_wide.zarr}"
ALIGNMENT_REPORT="${STATS_ALIGNMENT_REPORT:-data/processed/alignment_qc.json}"
OUTPUT="${STATS_OUT:-data/processed/stats.json}"
TRAIN_START="${STATS_TRAIN_START:-1981}"
TRAIN_END="${STATS_TRAIN_END:-2018}"
SAMPLE_DAYS="${STATS_SAMPLE_DAYS:-1500}"
TRANSFORM="${STATS_TRANSFORM:-log1p}"

[[ -d "$ZARR_STORE" ]] || {
    echo "ERROR: Packed Zarr store not found: $ZARR_STORE"
    exit 1
}
[[ -f "$ALIGNMENT_REPORT" ]] || {
    echo "ERROR: Alignment report not found: $ALIGNMENT_REPORT"
    exit 1
}

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
unset PYTHONHOME
unset PYTHONPATH

if [[ -n "${STATS_PYTHON:-}" ]]; then
    PYTHON_BIN="$STATS_PYTHON"
else
    PYTHON_BIN=""
    PYTHON_CANDIDATES=(
        "$SLURM_SUBMIT_DIR/../envs/bdda-gh200/bin/python"
        "/home/afahad/nb/project/BDDA/envs/bdda-gh200/bin/python"
    )
    for candidate in "${PYTHON_CANDIDATES[@]}"; do
        if [[ -x "$candidate" ]]; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
    [[ -n "$PYTHON_BIN" ]] || {
        echo "ERROR: ARM-native project Python was not found."
        echo "Set STATS_PYTHON=/absolute/path/to/ARM/python and resubmit."
        exit 1
    }
fi

"$PYTHON_BIN" - "$ALIGNMENT_REPORT" <<'PY'
import json
import sys

import numpy
import zarr

report = json.load(open(sys.argv[1]))
assert report.get("passed") is True, "daily alignment report did not pass"
print("numpy:", numpy.__version__)
print("zarr:", zarr.__version__)
print("alignment peak lag:", report["peak_lag_days"])
PY

mkdir -p "$(dirname "$OUTPUT")"

echo "Repository: $SLURM_SUBMIT_DIR"
echo "Architecture: $(uname -m)"
echo "Packed store: $ZARR_STORE"
echo "Training years: $TRAIN_START-$TRAIN_END"
echo "Sample days: $SAMPLE_DAYS"
echo "Transform: $TRANSFORM"
echo "Output: $OUTPUT"
echo "Python: $PYTHON_BIN"

STATS_ARGS=()
# STATS_RESIDUAL=1 parameterises the target as a correction to ERA5 tp in
# transformed space rather than the absolute field.
if [[ "${STATS_RESIDUAL:-0}" == "1" ]]; then
    STATS_ARGS+=(--residual --era5-tp-index "${STATS_ERA5_TP_INDEX:-0}")
    echo "Target: RESIDUAL (base = conditioning channel ${STATS_ERA5_TP_INDEX:-0})"
else
    echo "Target: ABSOLUTE"
fi

srun --ntasks=1 "$PYTHON_BIN" -u scripts/06_compute_stats.py \
    --zarr "$ZARR_STORE" \
    --train-years "$TRAIN_START" "$TRAIN_END" \
    --transform "$TRANSFORM" \
    --sample-days "$SAMPLE_DAYS" \
    "${STATS_ARGS[@]}" \
    --out "$OUTPUT"
