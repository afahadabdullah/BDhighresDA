#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v3-tests
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

cd "${SLURM_SUBMIT_DIR:?submit this job with sbatch from the repository root}"
export PYTHONPATH="$PWD/src"
mkdir -p logs

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: V3 tests must run on an aarch64 Grace node" >&2
    exit 1
}
[[ -x "$PYTHON_BIN" ]] || {
    echo "ERROR: ARM Python not found: $PYTHON_BIN" >&2
    exit 1
}

"$PYTHON_BIN" -m pytest -q \
    tests/test_v3_subgrid.py \
    tests/test_v4_da_test.py
