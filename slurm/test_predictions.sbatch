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

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: This job must run on an aarch64 Grace node."
    exit 1
}

CKPT="${TEST_CKPT:-runs/prior_h100/best.pt}"
CONFIG="${TEST_CONFIG:-configs/da.yaml}"
MEMBERS="${TEST_MEMBERS:-16}"
QUANTILES="${TEST_QUANTILES:-0.50 0.90 0.99}"

# Validate the paths ACTUALLY IN USE, after the variables exist.  This guard
# used to sit above these assignments and hardcode runs/prior_h100/best.pt and
# data/processed/stats.json, so it rejected every checkpoint except the v1 one
# and reported it as a single unhelpful line.  Each failure now names the file.
STATS_FILE="${TEST_STATS:-$("$PYTHON_BIN" -c "
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))['data']['stats'])
" "$CONFIG" 2>/dev/null)}"

missing=0
for path in scripts/08_plot_test_predictions.py "$CONFIG" "$CKPT" "$STATS_FILE"; do
    [[ -f "$path" ]] || { echo "ERROR: missing file: $path"; missing=1; }
done
[[ -d data/processed/bd_wide.zarr ]] || {
    echo "ERROR: missing store: data/processed/bd_wide.zarr"; missing=1
}
[[ "$missing" -eq 0 ]] || exit 1
echo "Statistics: $STATS_FILE"
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

[[ -f "$CKPT" ]] || {
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
}

echo "Checkpoint: $CKPT"
echo "Config: $CONFIG"
echo "Members: $MEMBERS"
echo "Figure: $FIGURE"
echo "Metrics figure: $METRICS_FIGURE"
echo "Individual case figures: $CASE_DIR"
echo "Report: $REPORT"
echo "Cartopy data: $CARTOPY_DIR"

EXTRA=()
[[ -n "${TEST_STATS:-}" ]] && EXTRA+=(--stats "$TEST_STATS")
[[ -n "${TEST_CFG_SCALE:-}" ]] && EXTRA+=(--cfg-scale "$TEST_CFG_SCALE")

srun --ntasks=1 "$PYTHON_BIN" -u scripts/08_plot_test_predictions.py \
    --config "$CONFIG" \
    --quantiles $QUANTILES \
    "${EXTRA[@]}" \
    --ckpt "$CKPT" \
    --members "$MEMBERS" \
    --out-figure "$FIGURE" \
    --out-metrics-figure "$METRICS_FIGURE" \
    --out-case-dir "$CASE_DIR" \
    --out-report "$REPORT" \
    --cartopy-data-dir "$CARTOPY_DIR"
