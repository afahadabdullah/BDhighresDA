#!/usr/bin/env bash
#SBATCH --job-name=cpc-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# The login nodes are x86_64 while grace-cpuonly jobs can run on aarch64 Grace
# nodes.  Never inherit the login-node Miniforge python through PATH.
module purge 2>/dev/null || true

cd "$SLURM_SUBMIT_DIR"

START="${CPC_START:-1981}"
END="${CPC_END:-2025}"
OUT_DIR="${CPC_OUT:-data/raw/cpc}"

export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

if [[ -n "${CPC_PYTHON:-}" ]]; then
    PYTHON_BIN="$CPC_PYTHON"
elif [[ "$(uname -m)" == "aarch64" ]]; then
    PYTHON_BIN=""
    ARM_PYTHON_CANDIDATES=(
        "$SLURM_SUBMIT_DIR/../envs/bdda-gh200/bin/python"
        "/home/afahad/nb/project/BDDA/envs/bdda-gh200/bin/python"
    )
    for candidate in "${ARM_PYTHON_CANDIDATES[@]}"; do
        if [[ -x "$candidate" ]]; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
    [[ -n "$PYTHON_BIN" ]] || {
        echo "ERROR: ARM-native project Python was not found."
        echo "Set CPC_PYTHON=/absolute/path/to/ARM/python and resubmit."
        exit 1
    }
else
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi

[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || {
    echo "ERROR: Python executable not found: ${PYTHON_BIN:-unset}"
    exit 1
}

[[ -f scripts/02b_download_cpc.py ]] || {
    echo "ERROR: Submit from the BDhighresDA repository root."
    exit 1
}

mkdir -p "$OUT_DIR"
echo "Architecture: $(uname -m)"
echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" -c 'import platform, sys; print("Python architecture:", platform.machine()); print("Python executable:", sys.executable)'
echo "Downloading complete CPC record $START-$END to $OUT_DIR"
srun --ntasks=1 "$PYTHON_BIN" -u scripts/02b_download_cpc.py \
    --start "$START" \
    --end "$END" \
    --out "$OUT_DIR" \
    --require-complete
