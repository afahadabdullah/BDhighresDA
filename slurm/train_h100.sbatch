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

export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"
PYTHON_BIN="$ENV_PREFIX/bin/python"

export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || {
    echo "ERROR: ARM training Python not found: $PYTHON_BIN"
    exit 1
}

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

"$PYTHON_BIN" - <<'PY'
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

PREFLIGHT_REPORT="${TRAIN_PREFLIGHT_REPORT:-data/processed/training_preflight.json}"
if [[ "${REQUIRE_TRAINING_PREFLIGHT:-1}" == "1" ]]; then
    [[ -f "$PREFLIGHT_REPORT" ]] || {
        echo "ERROR: GH200 real-data preflight report not found: $PREFLIGHT_REPORT"
        echo "Run slurm/submit_preflight_training_gh200.sh first."
        exit 1
    }
    "$PYTHON_BIN" - "$PREFLIGHT_REPORT" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
assert report.get("passed") is True, "training preflight did not pass"
head = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
assert report["git_commit"] == head, (
    "repository changed after preflight; rerun the GH200 preflight"
)
for path_key, hash_key in (
    ("config", "config_sha256"),
    ("stats", "stats_sha256"),
    ("normalization_report", "normalization_report_sha256"),
    ("normalization_figure", "normalization_figure_sha256"),
):
    path = Path(report[path_key])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == report[hash_key], (
        f"{path} changed after preflight; rerun the GH200 preflight"
    )
print("Verified GH200 preflight:", sys.argv[1])
print("Preflight validation loss:", report["validation_loss"])
print("Preflight peak reserved GiB:", report["peak_memory_reserved_gib"])
PY
fi

TRAIN_ARGS=()

if [[ "${RESUME_IF_AVAILABLE:-1}" == "1" &&
      -f runs/prior_h100/last.pt ]]; then
    echo "Resuming from runs/prior_h100/last.pt"
    TRAIN_ARGS+=(--resume runs/prior_h100/last.pt)
fi

srun --ntasks=1 "$PYTHON_BIN" -u scripts/train.py \
    --config configs/train_h100.yaml \
    "${TRAIN_ARGS[@]}"
