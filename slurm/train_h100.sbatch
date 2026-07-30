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

# The preflight gate is now OPT-IN.  Set REQUIRE_TRAINING_PREFLIGHT=1 to enforce
# it.  Rationale: over the v2-v4 runs this gate blocked submission three times,
# every time because a commit had moved, and it never once caught a real problem.
# The preflight JOB is still worth running by hand -- two real optimiser steps on
# real data is a cheap smoke test before a long allocation -- but pinning the
# commit made routine code edits require rebuilding a two-job chain.
PREFLIGHT_REPORT="${TRAIN_PREFLIGHT_REPORT:-data/processed/training_preflight.json}"
if [[ "${REQUIRE_TRAINING_PREFLIGHT:-0}" == "1" ]]; then
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

import os

strict = os.environ.get("PREFLIGHT_PIN_COMMIT", "0") == "1"
report = json.loads(Path(sys.argv[1]).read_text())
assert report.get("passed") is True, "training preflight did not pass"
head = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if report["git_commit"] != head:
    message = (
        f"repository moved since the preflight "
        f"({report['git_commit'][:8]} -> {head[:8]})"
    )
    if strict:
        raise AssertionError(message + "; rerun the GH200 preflight")
    print(f"WARNING: {message}; continuing (PREFLIGHT_PIN_COMMIT != 1)")
for path_key, hash_key in (
    ("config", "config_sha256"),
    ("stats", "stats_sha256"),
    ("normalization_report", "normalization_report_sha256"),
    ("normalization_figure", "normalization_figure_sha256"),
):
    path = Path(report[path_key])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != report[hash_key]:
        message = f"{path} changed after the preflight"
        if strict:
            raise AssertionError(message + "; rerun the GH200 preflight")
        print(f"WARNING: {message}; continuing (PREFLIGHT_PIN_COMMIT != 1)")
print("Verified GH200 preflight:", sys.argv[1])
print("Preflight validation loss:", report["validation_loss"])
print("Preflight peak reserved GiB:", report["peak_memory_reserved_gib"])
PY
fi

TRAIN_ARGS=()

CONFIG="${CONFIG:-configs/train_h100.yaml}"

# Read out_dir from the config rather than hardcoding it.  The old hardcoded
# runs/prior_h100 path was a live footgun: after the conditioning-transform
# change, a stale last.pt in that directory would have been silently resumed
# into a run whose input channels no longer mean the same thing.
OUT_DIR="$("$PYTHON_BIN" -c "
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))['train']['out_dir'])
" "$CONFIG")"
echo "Config:  $CONFIG"
echo "Out dir: $OUT_DIR"

# Guard against resuming across a change in input semantics: the checkpoint
# records the stats file it was trained with.
if [[ -f "$OUT_DIR/last.pt" ]]; then
    "$PYTHON_BIN" - "$OUT_DIR/last.pt" "$CONFIG" <<'PY'
import sys
import torch
import yaml

ckpt = torch.load(sys.argv[1], map_location="cpu")
want = yaml.safe_load(open(sys.argv[2]))["data"]["stats"]
have = ckpt.get("cfg", {}).get("data", {}).get("stats")
if have != want:
    raise SystemExit(
        f"REFUSING TO RESUME: {sys.argv[1]} was trained with stats={have!r} but "
        f"the config asks for {want!r}. The conditioning transform differs, so "
        f"the weights expect different input distributions. Use a fresh "
        f"train.out_dir, or set RESUME_IF_AVAILABLE=0 to start from scratch."
    )
print(f"Resume check passed: stats={have!r}")
PY
fi

if [[ "${RESUME_IF_AVAILABLE:-1}" == "1" && -f "$OUT_DIR/last.pt" ]]; then
    echo "Resuming from $OUT_DIR/last.pt"
    TRAIN_ARGS+=(--resume "$OUT_DIR/last.pt")
else
    echo "Starting from scratch (no checkpoint, or RESUME_IF_AVAILABLE=0)"
fi

srun --ntasks=1 "$PYTHON_BIN" -u scripts/train.py \
    --config "$CONFIG" \
    "${TRAIN_ARGS[@]}"
