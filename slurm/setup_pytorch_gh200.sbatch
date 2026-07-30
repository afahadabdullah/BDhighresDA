#!/usr/bin/env bash
#SBATCH --job-name=pytorch-setup
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module purge 2>/dev/null || true

ENV_PREFIX="${GH200_ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="$ENV_PREFIX/bin/python"
TORCH_VERSION="${GH200_TORCH_VERSION:-2.9.1}"
TORCH_INDEX="${GH200_TORCH_INDEX:-https://download.pytorch.org/whl/cu129}"

export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: PyTorch must be installed from an aarch64 Grace node."
    exit 1
}
[[ -x "$PYTHON_BIN" ]] || {
    echo "ERROR: Existing ARM project environment not found: $ENV_PREFIX"
    exit 1
}

echo "Environment: $ENV_PREFIX"
echo "Python: $PYTHON_BIN"
echo "Architecture: $(uname -m)"
echo "Requested PyTorch: $TORCH_VERSION"
echo "Wheel index: $TORCH_INDEX"
"$PYTHON_BIN" --version

if "$PYTHON_BIN" - <<'PY'
import torch

assert torch.version.cuda is not None
print("Existing GPU-enabled PyTorch:", torch.__version__)
print("Existing PyTorch CUDA build:", torch.version.cuda)
PY
then
    echo "A GPU-enabled PyTorch installation already exists; keeping it."
else
    echo "Installing the NCCS-recommended Grace Hopper PyTorch wheel."
    "$PYTHON_BIN" -m pip install \
        --upgrade \
        "torch==$TORCH_VERSION" \
        --index-url "$TORCH_INDEX"
fi

"$PYTHON_BIN" - <<'PY'
import platform
import sys

import torch

print("Python:", sys.executable)
print("Architecture:", platform.machine())
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
assert platform.machine() == "aarch64"
assert "/envs/bdda-gh200/" in sys.executable
assert torch.version.cuda is not None
assert torch.cuda.is_available(), "PyTorch cannot access the allocated GH200"
major, minor = torch.cuda.get_device_capability(0)
print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", f"{major}.{minor}")
assert major >= 9, "allocated GPU is not Hopper-class"
PY

echo "PYTORCH SETUP PASSED"
