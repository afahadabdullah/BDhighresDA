#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v7-train
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || {
    echo "ERROR: ARM Python not found: $PYTHON_BIN" >&2
    exit 1
}

cd "${SLURM_SUBMIT_DIR:?submit this job with sbatch from the repository root}"
export PYTHONPATH="$PWD/src"
mkdir -p logs

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: V7 training must run on an aarch64 Grace/GH200 node" >&2
    exit 1
}
[[ -f scripts/57_train_subgrid_oracle.py ]] || {
    echo "ERROR: submit from the BDhighresDA repository root" >&2
    exit 1
}

CONFIG="${V7_CONFIG:?set V7_CONFIG to one V3 phase YAML}"
EXPECTED_STAGE="${V7_EXPECTED_STAGE:-}"
[[ -f "$CONFIG" ]] || { echo "ERROR: missing config: $CONFIG" >&2; exit 1; }

mapfile -t META < <("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg["stage"])
print(cfg["train"]["out_dir"])
print(cfg["data"]["zarr"])
PY
)
STAGE="${META[0]}"
OUT_DIR="${META[1]}"
TARGET_STORE="${META[2]}"

if [[ -n "$EXPECTED_STAGE" && "$STAGE" != "$EXPECTED_STAGE" ]]; then
    echo "ERROR: config stage $STAGE differs from expected $EXPECTED_STAGE" >&2
    exit 1
fi

"$PYTHON_BIN" - "$CONFIG" "$TARGET_STORE" "$OUT_DIR" <<'PY'
import sys
from pathlib import Path

import torch
import yaml
import zarr

from bdhires.data import SUBGRID_SCHEMA

config_path, target_path, output_path = sys.argv[1:]
cfg = yaml.safe_load(open(config_path))
target = zarr.open_group(target_path, mode="r")
assert target.attrs.get("complete", False), f"incomplete target archive: {target_path}"
# Take the expected schema from the library instead of pinning it here.  A bump
# this preflight does not follow would train a new model against a stale archive,
# which is the exact failure the bump exists to prevent.
assert target.attrs.get("schema") == SUBGRID_SCHEMA, (
    f"target archive schema {target.attrs.get('schema')!r} is not {SUBGRID_SCHEMA!r}; "
    "rebuild it with scripts/56_build_chirps_subgrid_targets.py"
)
target_encoding = dict(target.attrs["subgrid_encoding"])

# Tiled validation covers the domain with non-overlapping crops, so the crop has
# to divide it.  The dataset enforces this, but only once the training job has
# started and loaded data -- by which point the archive build and the queue wait
# are already spent.  Check it here, in seconds, instead.
crop = int(cfg["data"]["crop"])
height, width = target["fine_mm"].shape[-2:]
for name, extent in (("height", height), ("width", width)):
    assert extent % crop == 0, (
        f"crop {crop} does not divide the domain {name} {extent}; tiled validation "
        "would overlap and weight some cells twice"
    )
levels = len(
    cfg["model"].get("channel_mult", cfg["model"].get("allocation", {}).get("channel_mult", [1]))
) - 1
side = crop if cfg["stage"] != "coarse" else crop // int(cfg["data"].get("factor", 2))
assert side % 2**levels == 0, (
    f"stage {cfg['stage']} runs at {side} cells, which does not survive {levels} downsamplings"
)

output = Path(output_path)
last = output / "last.pt"
if last.exists():
    checkpoint = torch.load(last, map_location="cpu")
    assert checkpoint.get("schema") == SUBGRID_SCHEMA
    assert checkpoint.get("stage") == cfg["stage"]
    assert dict(checkpoint.get("subgrid_encoding", {})) == target_encoding, (
        "resume checkpoint encoding differs from the target archive"
    )
    assert checkpoint.get("config") == cfg, (
        "refusing to resume because the resolved config differs from the checkpoint"
    )
elif output.exists() and any(output.iterdir()):
    raise RuntimeError(
        f"{output} contains partial files but no last.pt; inspect or move it aside"
    )

if cfg["stage"] == "joint":
    for key, expected_stage in (
        ("init_coarse", "coarse"),
        ("init_allocation", "allocation"),
    ):
        path = Path(cfg["train"][key])
        assert path.is_file(), f"missing joint initializer: {path}"
        checkpoint = torch.load(path, map_location="cpu")
        assert checkpoint.get("schema") == SUBGRID_SCHEMA, (
            f"incompatible joint initializer schema: {path}"
        )
        assert checkpoint.get("stage") == expected_stage, (
            f"wrong joint initializer stage: {path}"
        )
        assert dict(checkpoint.get("subgrid_encoding", {})) == target_encoding, (
            f"joint initializer encoding differs from target: {path}"
        )

print("Preflight passed")
print("  stage:", cfg["stage"])
print("  target:", target_path)
print("  output:", output)
PY

"$PYTHON_BIN" - <<'PY'
import platform
import sys
import torch

print("Python:", sys.executable)
print("Architecture:", platform.machine())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
assert platform.machine() == "aarch64"
assert torch.cuda.is_available(), "PyTorch cannot access the allocated GPU"
print("GPU:", torch.cuda.get_device_name(0))
PY

TRAIN_ARGS=(--config "$CONFIG")
if [[ -f "$OUT_DIR/last.pt" ]]; then
    TRAIN_ARGS+=(--resume "$OUT_DIR/last.pt")
    echo "Resuming $STAGE from $OUT_DIR/last.pt"
else
    echo "Starting $STAGE from scratch"
fi

srun --ntasks=1 "$PYTHON_BIN" -u scripts/57_train_subgrid_oracle.py \
    "${TRAIN_ARGS[@]}"

[[ -f "$OUT_DIR/last.pt" && -f "$OUT_DIR/best.pt" ]] || {
    echo "ERROR: $STAGE exited without both last.pt and best.pt" >&2
    exit 1
}
echo "V7 $STAGE stage complete: $OUT_DIR/best.pt"
