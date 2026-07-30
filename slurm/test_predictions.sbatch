#!/usr/bin/env bash
#SBATCH --job-name=test-predictions
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module purge 2>/dev/null || true

export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"
PYTHON_BIN="$ENV_PREFIX/bin/python"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/${USER}/bdda-test-matplotlib-${SLURM_JOB_ID}"
unset PYTHONHOME
unset PYTHONPATH
mkdir -p "$MPLCONFIGDIR"

[[ -x "$PYTHON_BIN" ]] || {
    echo "ERROR: ARM training Python not found: $PYTHON_BIN"
    exit 1
}

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"

[[ -f scripts/08_plot_test_predictions.py &&
   -f configs/da.yaml &&
   -f runs/prior_h100/best.pt &&
   -d data/processed/bd_wide.zarr &&
   -f data/processed/stats.json ]] || {
    echo "ERROR: Test script, best checkpoint, data, config, or statistics are missing."
    exit 1
}

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: This job must run on an aarch64 Grace node."
    exit 1
}

MEMBERS="${TEST_MEMBERS:-16}"
FIGURE="${TEST_MAP_FIGURE:-data/processed/test_prediction_maps.png}"
METRICS_FIGURE="${TEST_METRICS_FIGURE:-data/processed/test_prediction_metrics.png}"
CASE_DIR="${TEST_CASE_DIR:-data/processed/test_prediction_cases}"
REPORT="${TEST_REPORT:-data/processed/test_prediction_report.json}"
CARTOPY_DIR="${CARTOPY_DATA_DIR:-data/static/cartopy}"
mkdir -p "$CARTOPY_DIR"

"$PYTHON_BIN" - <<'PY'
import cartopy
import matplotlib

print("Cartopy:", cartopy.__version__)
print("Matplotlib:", matplotlib.__version__)
PY

echo "Checkpoint: runs/prior_h100/best.pt"
echo "Members: $MEMBERS"
echo "Figure: $FIGURE"
echo "Metrics figure: $METRICS_FIGURE"
echo "Individual case figures: $CASE_DIR"
echo "Report: $REPORT"
echo "Cartopy data: $CARTOPY_DIR"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/08_plot_test_predictions.py \
    --config configs/da.yaml \
    --ckpt runs/prior_h100/best.pt \
    --members "$MEMBERS" \
    --out-figure "$FIGURE" \
    --out-metrics-figure "$METRICS_FIGURE" \
    --out-case-dir "$CASE_DIR" \
    --out-report "$REPORT" \
    --cartopy-data-dir "$CARTOPY_DIR"
