#!/bin/bash
#SBATCH --job-name=bdhires-tests
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

# Step 0 of docs/RUNBOOK_v2.md.
#
# These tests cannot run on the login node: bdda-gh200 is an aarch64 build and
# gpulogin1 is x86_64, so invoking its python there fails with
# "cannot execute binary file: Exec format error".  The test suite itself is
# CPU-only, but the aarch64 interpreter only exists on the Grace partition.

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

[[ -f tests/test_conditioning_fixes.py ]] || {
    echo "ERROR: Submit this job from the BDhighresDA repository root."
    exit 1
}

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: This job must run on an aarch64 Grace node."
    exit 1
}

"$PYTHON_BIN" -c "
import platform, sys, torch
print('Python:      ', sys.executable)
print('Architecture:', platform.machine())
print('PyTorch:     ', torch.__version__)
print('CUDA avail:  ', torch.cuda.is_available())
assert platform.machine() == 'aarch64'
"

echo
echo "=============== pytest: required regression suite ==============="
"$PYTHON_BIN" -m pytest \
    tests/test_conditioning_fixes.py \
    tests/test_progress_and_summary.py \
    tests/test_scale_metrics.py \
    tests/test_osse_downscaling.py \
    -v --tb=short

echo
echo "=============== pytest: full suite ==============="
# Not fatal: some modules need xarray/rasterio that may not be in this env.
"$PYTHON_BIN" -m pytest tests/ -q --tb=short || {
    echo "NOTE: full-suite failures above may be pre-existing dependency gaps."
    echo "      The v2 suite result printed earlier is the one that matters."
}

echo
echo "=============== smoke test ==============="
"$PYTHON_BIN" scripts/smoke_test.py

echo
echo "STEP 0 COMPLETE"
