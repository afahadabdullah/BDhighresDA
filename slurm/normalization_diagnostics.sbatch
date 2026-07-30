#!/usr/bin/env bash
#SBATCH --job-name=norm-diagnostics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

ZARR_STORE="${NORM_ZARR:-data/processed/bd_wide.zarr}"
STATS_FILE="${NORM_STATS:-data/processed/stats.json}"
FIGURE="${NORM_FIGURE:-data/processed/normalization_diagnostics.png}"
REPORT="${NORM_REPORT:-data/processed/normalization_diagnostics.json}"
SAMPLE_DAYS="${NORM_SAMPLE_DAYS:-256}"
PIXELS_PER_DAY="${NORM_PIXELS_PER_DAY:-2048}"

[[ -f scripts/06_plot_normalization.py &&
   -d "$ZARR_STORE" &&
   -f "$STATS_FILE" ]] || {
    echo "ERROR: Normalization script, packed data, or statistics are missing."
    exit 1
}

export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/${USER}/bdda-matplotlib-${SLURM_JOB_ID}"
unset PYTHONHOME
unset PYTHONPATH
mkdir -p "$MPLCONFIGDIR"

if [[ -n "${NORM_PYTHON:-}" ]]; then
    PYTHON_BIN="$NORM_PYTHON"
else
    PYTHON_BIN="/home/afahad/nb/project/BDDA/envs/bdda-gh200/bin/python"
fi
[[ -x "$PYTHON_BIN" ]] || {
    echo "ERROR: ARM project Python not found: $PYTHON_BIN"
    exit 1
}

"$PYTHON_BIN" - <<'PY'
import matplotlib
import numpy
import zarr

print("matplotlib:", matplotlib.__version__)
print("numpy:", numpy.__version__)
print("zarr:", zarr.__version__)
PY

mkdir -p "$(dirname "$FIGURE")" "$(dirname "$REPORT")"

echo "Repository: $SLURM_SUBMIT_DIR"
echo "Architecture: $(uname -m)"
echo "Packed store: $ZARR_STORE"
echo "Statistics: $STATS_FILE"
echo "Figure: $FIGURE"
echo "Report: $REPORT"
echo "Python: $PYTHON_BIN"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/06_plot_normalization.py \
    --zarr "$ZARR_STORE" \
    --stats "$STATS_FILE" \
    --out "$FIGURE" \
    --report "$REPORT" \
    --sample-days "$SAMPLE_DAYS" \
    --pixels-per-day "$PIXELS_PER_DAY"
