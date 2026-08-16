#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v3-legacy-diag
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

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || {
    echo "ERROR: ARM Python not found: $PYTHON_BIN" >&2
    exit 1
}
cd "${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
export PYTHONPATH="$PWD/src"
mkdir -p logs

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: legacy V3 diagnostic must run on a Grace/GH200 node" >&2
    exit 1
}

START="${LEGACY_V3_START:-2022-05-01}"
END="${LEGACY_V3_END:-2022-05-05}"
MEMBERS="${LEGACY_V3_MEMBERS:-4}"
STEPS="${LEGACY_V3_STEPS:-25}"
OFFSET="${LEGACY_V3_BACKGROUND_OFFSET:--1}"
GAUGE_SIGMA="${LEGACY_V3_GAUGE_SIGMA_MM:-3.0}"
GUIDANCE_GAMMA="${LEGACY_V3_GUIDANCE_GAMMA:-1.0}"
GUIDANCE_SCALE="${LEGACY_V3_GUIDANCE_SCALE:-1.0}"
GUIDANCE_CLIP="${LEGACY_V3_GUIDANCE_CLIP_NORM:-100.0}"
HUBER_DELTA="${LEGACY_V3_HUBER_DELTA:-3.0}"
CHECKPOINT="${LEGACY_V3_CHECKPOINT:-runs/prior_h100_cpc_v3_subgrid/joint/best.pt}"
TARGET="${LEGACY_V3_TARGET:-data/processed/cpc_v3_subgrid/wide_cpc.zarr}"
DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
STATION_CATALOG="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
OUT_DIR="${LEGACY_V3_OUT_DIR:-data/processed/v3_legacy_diagnostic/may2022_5day}"
DAILY="$OUT_DIR/bmd_daily.csv"
STATION_SUMMARY="$OUT_DIR/bmd_stations.csv"
BMD_QC="$OUT_DIR/bmd_qc.json"

for required in \
    scripts/05_convert_bmd_dir.py \
    scripts/59_legacy_v3_subgrid_diagnostic.py \
    "$CHECKPOINT" \
    "$STATION_CATALOG"; do
    [[ -e "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done
[[ -d "$TARGET" ]] || { echo "ERROR: missing target store $TARGET" >&2; exit 1; }
[[ -d "$DATA_DIR" ]] || { echo "ERROR: missing BMD directory $DATA_DIR" >&2; exit 1; }
mkdir -p "$OUT_DIR"

"$PYTHON_BIN" - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch cannot access the allocated GPU"
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
PY

"$PYTHON_BIN" -u scripts/59_legacy_v3_subgrid_diagnostic.py \
    --target-store "$TARGET" \
    --checkpoint "$CHECKPOINT" \
    --stations "$DAILY" \
    --preflight-only

if [[ ! -s "$DAILY" ]]; then
    "$PYTHON_BIN" -u scripts/05_convert_bmd_dir.py \
        --data-dir "$DATA_DIR" \
        --stations "$STATION_CATALOG" \
        --start "$START" --end "$END" \
        --out "$DAILY" \
        --summary "$STATION_SUMMARY" \
        --report "$BMD_QC"
else
    echo "Reusing canonical BMD file: $DAILY"
fi

echo "Legacy pre-v4 V3-SG diagnostic"
echo "  observation dates: $START through $END"
echo "  condition offset:  $OFFSET day(s)"
echo "  ensemble:          $MEMBERS members; $STEPS Heun steps"
echo "  gauge likelihood:  sigma=$GAUGE_SIGMA mm, gamma=$GUIDANCE_GAMMA, scale=$GUIDANCE_SCALE"
echo "  checkpoint:        $CHECKPOINT"
echo "  target:            $TARGET"
echo "  output:            $OUT_DIR"
echo "  WARNING: diagnostic only; do not pool with corrected v4"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/59_legacy_v3_subgrid_diagnostic.py \
    --target-store "$TARGET" \
    --checkpoint "$CHECKPOINT" \
    --stations "$DAILY" \
    --start "$START" --end "$END" \
    --background-day-offset "$OFFSET" \
    --members "$MEMBERS" \
    --n-steps "$STEPS" \
    --gauge-sigma-mm "$GAUGE_SIGMA" \
    --guidance-gamma "$GUIDANCE_GAMMA" \
    --guidance-scale "$GUIDANCE_SCALE" \
    --guidance-clip-norm "$GUIDANCE_CLIP" \
    --huber-delta "$HUBER_DELTA" \
    --out-dir "$OUT_DIR"

for result in \
    "$OUT_DIR/legacy_v3_may2022_5day.zarr" \
    "$OUT_DIR/legacy_v3_may2022_5day.json" \
    "$OUT_DIR/legacy_v3_may2022_5day.png" \
    "$OUT_DIR/legacy_v3_may2022_5day_subgrid.png"; do
    [[ -e "$result" ]] || { echo "ERROR: expected output missing: $result" >&2; exit 1; }
done
echo "Legacy diagnostic complete: $OUT_DIR"
