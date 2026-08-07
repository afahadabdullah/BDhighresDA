#!/usr/bin/env bash
#SBATCH --job-name=training-preflight
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module purge 2>/dev/null || true

export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"
PYTHON_BIN="$ENV_PREFIX/bin/python"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || {
    echo "ERROR: ARM training Python not found: $PYTHON_BIN"
    exit 1
}

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"

PREFLIGHT_CONFIG="${PREFLIGHT_CONFIG:-configs/train_h100.yaml}"
NORM_REPORT="${PREFLIGHT_NORM_REPORT:-data/processed/normalization_diagnostics.json}"
ZARR_STORE="$("$PYTHON_BIN" -c "
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))['data']['zarr'])
" "$PREFLIGHT_CONFIG")"
# The statistics file is whatever the config asks for -- do NOT hardcode
# stats.json here.  The v2 conditioning transform lives in stats_v2.json, and a
# hardcoded check would happily pass on the stale v1 file while training read
# the other one.
STATS_FILE="$("$PYTHON_BIN" -c "
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))['data']['stats'])
" "$PREFLIGHT_CONFIG")"

# Name the missing item. The old message listed four possibilities and left the
# caller to guess, which is a poor trade when each guess costs a queue slot.
MISSING=()
[[ -f scripts/preflight_training.py ]] || MISSING+=("scripts/preflight_training.py")
[[ -f "$PREFLIGHT_CONFIG" ]]          || MISSING+=("config: $PREFLIGHT_CONFIG")
[[ -d "$ZARR_STORE" ]]                || MISSING+=("zarr store: $ZARR_STORE")
[[ -f "$NORM_REPORT" ]]               || MISSING+=("normalization report: $NORM_REPORT
      It depends only on the zarr and the stats, so one report covers every
      config sharing them. Generate it with:
        python scripts/06_plot_normalization.py \\
            --zarr $ZARR_STORE --stats $STATS_FILE \\
            --out-json $NORM_REPORT --out-figure ${NORM_REPORT%.json}.png")

[[ ${#MISSING[@]} -eq 0 ]] || {
    echo "ERROR: preflight cannot run. Missing:"
    for item in "${MISSING[@]}"; do echo "  - $item"; done
    exit 1
}

echo "Config: $PREFLIGHT_CONFIG"
echo "Statistics: $STATS_FILE"
[[ -f "$STATS_FILE" ]] || {
    echo "ERROR: statistics file named by $PREFLIGHT_CONFIG not found: $STATS_FILE"
    echo "Run scripts/06_compute_stats.py with --out $STATS_FILE first."
    exit 1
}

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: This job must run on an aarch64 Grace node."
    exit 1
}

"$PYTHON_BIN" - <<'PY'
import platform
import sys

import scipy
import torch
import zarr

print("Python:", sys.executable)
print("Architecture:", platform.machine())
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("Zarr:", zarr.__version__)
print("SciPy:", scipy.__file__)
print("CUDA available:", torch.cuda.is_available())

assert platform.machine() == "aarch64"
assert "/envs/bdda-gh200/" in sys.executable
assert "/.local/" not in scipy.__file__
assert torch.cuda.is_available(), "PyTorch cannot access the allocated GPU"

print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU memory GiB:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
)
PY

STEPS="${PREFLIGHT_STEPS:-2}"
OUTPUT="${PREFLIGHT_OUT:-data/processed/training_preflight.json}"

echo "Repository: $SLURM_SUBMIT_DIR"
echo "Preflight steps: $STEPS"
echo "Report: $OUTPUT"

PREFLIGHT_ARGS=()
# Off by default; set PREFLIGHT_PIN_COMMIT=1 to fail when the repo has moved
# since the normalization diagnostics ran.
if [[ "${PREFLIGHT_PIN_COMMIT:-0}" == "1" ]]; then
    PREFLIGHT_ARGS+=(--strict-commit)
fi

srun --ntasks=1 "$PYTHON_BIN" -u scripts/preflight_training.py \
    --config "$PREFLIGHT_CONFIG" \
    --steps "$STEPS" \
    --normalization-report "$NORM_REPORT" \
    "${PREFLIGHT_ARGS[@]}" \
    --out "$OUTPUT"
