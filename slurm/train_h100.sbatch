#!/bin/bash
#SBATCH --job-name=bdhires-gh200
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module purge 2>/dev/null || true

export ARM_CONDA="/home/afahad/nb/project/BDDA/miniforge3-aarch64"
export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"

export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

source "$ARM_CONDA/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# Submit from the repository root, preferably through submit_train_gh200.sh.
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"

[[ -f scripts/train.py && -f configs/train_h100.yaml ]] || {
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

TRAIN_ARGS=()

if [[ "${RESUME_IF_AVAILABLE:-1}" == "1" &&
      -f runs/prior_h100/last.pt ]]; then
    echo "Resuming from runs/prior_h100/last.pt"
    TRAIN_ARGS+=(--resume runs/prior_h100/last.pt)
fi

srun --ntasks=1 python -u scripts/train.py \
    --config configs/train_h100.yaml \
    "${TRAIN_ARGS[@]}"
