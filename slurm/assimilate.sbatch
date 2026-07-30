#!/bin/bash
#SBATCH --job-name=bdhires-da-gh200
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# Limit the array to two simultaneous GH200 jobs by default. Change %2 if needed.
#SBATCH --array=2020-2025%2
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

module purge 2>/dev/null || true

export ARM_CONDA="/home/afahad/nb/project/BDDA/miniforge3-aarch64"
export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"

export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

source "$ARM_CONDA/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# Submit from the repository root, preferably through submit_assimilate_gh200.sh.
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"

[[ -f scripts/assimilate.py && -f configs/da.yaml ]] || {
    echo "ERROR: Submit this job from the BDhighresDA repository root."
    exit 1
}

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: This job must run on an aarch64 Grace node."
    exit 1
}

python - <<'PY'
import platform
import sys

import scipy
import torch

print("Python:", sys.executable)
print("Architecture:", platform.machine())
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
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

[[ -f runs/prior_h100/best.pt ]] || {
    echo "ERROR: Missing runs/prior_h100/best.pt"
    exit 1
}

Y="$SLURM_ARRAY_TASK_ID"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/assimilate.py \
    --config configs/da.yaml \
    --ckpt runs/prior_h100/best.pt \
    --start "${Y}-01-01" \
    --end "${Y}-12-31" \
    --mode analysis \
    --out "data/processed/bdhires_analysis_${Y}.nc"
