#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v4-da-test
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

cd "${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
export PYTHONPATH="$PWD/src"
export MPLBACKEND=Agg
mkdir -p logs

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: the v4 test must run on a Grace/GH200 node" >&2
    exit 1
}
[[ -x "$PYTHON_BIN" ]] || {
    echo "ERROR: ARM Python not found: $PYTHON_BIN" >&2
    exit 1
}

START="${V4_TEST_START:-2022-05-01}"
END="${V4_TEST_END:-2022-05-05}"
MEMBERS="${V4_TEST_MEMBERS:-4}"
STEPS="${V4_TEST_STEPS:-25}"
OFFSET="${V4_TEST_BACKGROUND_OFFSET:--1}"
CHECKPOINT="${V4_TEST_CHECKPOINT:-runs/prior_h100_cpc_v3_subgrid_v4/joint/best.pt}"
TARGET="${V4_TEST_TARGET:-data/processed/cpc_v3_subgrid/wide_cpc_v4.zarr}"
IMERG="${V4_TEST_IMERG:-data/processed/imerg_prepared_ing2022/imerg_0p4deg_20220501_20220510.nc}"
OUT_ROOT="${V4_TEST_OUT_ROOT:-data/processed/v4_da_test/may2022_5day}"
RECOVER_INCOMPLETE="${V4_TEST_RECOVER_INCOMPLETE:-0}"
DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
STATION_CATALOG="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
DAILY="$OUT_ROOT/bmd_daily.csv"
STATION_SUMMARY="$OUT_ROOT/bmd_stations.csv"
BMD_QC="$OUT_ROOT/bmd_qc.json"
SAMPLE_STORE="$OUT_ROOT/v4_da_samples.zarr"
SAMPLE_REPORT="$OUT_ROOT/v4_da_sampling.json"
GRIDDED_DIR="$OUT_ROOT/evaluation/gridded"
EVALUATION_DIR="$OUT_ROOT/evaluation"
METHODS="background,gauges_withheld,imerg_only,simultaneous_withheld,gauges_all,simultaneous_all"

GAUGE_SIGMA="${V4_TEST_GAUGE_SIGMA_MM:-3.0}"
IMERG_SIGMA_FLOOR="${V4_TEST_IMERG_SIGMA_FLOOR_MM:-1.0}"
IMERG_REPRESENTATIVENESS="${V4_TEST_IMERG_REPRESENTATIVENESS_MM:-0.0}"
IMERG_R_MULTIPLIER="${V4_TEST_IMERG_R_MULTIPLIER:-1.0}"
GUIDANCE_GAMMA="${V4_TEST_GUIDANCE_GAMMA:-1.0}"
GUIDANCE_SCALE="${V4_TEST_GUIDANCE_SCALE:-1.0}"
GUIDANCE_CLIP="${V4_TEST_GUIDANCE_CLIP_NORM:-100.0}"
HUBER_DELTA="${V4_TEST_HUBER_DELTA:-3.0}"
WITHHOLD="${V4_TEST_WITHHOLD:-0.20}"
MIN_COVERAGE="${V4_TEST_MIN_COVERAGE:-0.80}"
HOLDOUT_NEIGHBOR_KM="${V4_TEST_HOLDOUT_NEIGHBOR_KM:-75.0}"
HOLDOUT_MAX_GAP_DEG="${V4_TEST_HOLDOUT_MAX_GAP_DEG:-200.0}"
SEED="${V4_TEST_SEED:-4202205}"

for required in \
    scripts/05_convert_bmd_dir.py \
    scripts/58_evaluate_subgrid_prior.py \
    scripts/60_v4_subgrid_da_test.py \
    scripts/61_evaluate_v4_subgrid_da_test.py \
    "$CHECKPOINT" \
    "$IMERG" \
    "$STATION_CATALOG"; do
    [[ -e "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done
[[ -d "$TARGET" ]] || { echo "ERROR: missing target store $TARGET" >&2; exit 1; }
[[ -d "$DATA_DIR" ]] || { echo "ERROR: missing BMD directory $DATA_DIR" >&2; exit 1; }
mkdir -p "$OUT_ROOT" "$GRIDDED_DIR" "$EVALUATION_DIR"

"$PYTHON_BIN" - <<'PY'
import platform
import torch

assert platform.machine() == "aarch64"
assert torch.cuda.is_available(), "PyTorch cannot access the allocated GPU"
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
PY

# This checks the corrected target/checkpoint decoder contract, requested dates,
# exact 0.4-degree footprint phase and IMERG error-correlation metadata before
# either BMD conversion or expensive sampling starts.
"$PYTHON_BIN" -u scripts/60_v4_subgrid_da_test.py \
    --target-store "$TARGET" \
    --checkpoint "$CHECKPOINT" \
    --stations "$DAILY" \
    --imerg "$IMERG" \
    --start "$START" --end "$END" \
    --background-day-offset "$OFFSET" \
    --members "$MEMBERS" --n-steps "$STEPS" \
    --withhold "$WITHHOLD" --min-coverage "$MIN_COVERAGE" \
    --holdout-neighbor-km "$HOLDOUT_NEIGHBOR_KM" \
    --holdout-max-gap-deg "$HOLDOUT_MAX_GAP_DEG" \
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

reuse_sample=0
recover_sample=0
if [[ -d "$SAMPLE_STORE" && -s "$SAMPLE_REPORT" ]]; then
    if "$PYTHON_BIN" - "$SAMPLE_STORE" "$SAMPLE_REPORT" "$TARGET" \
        "$CHECKPOINT" "$IMERG" "$START" "$END" "$MEMBERS" "$STEPS" \
        "$OFFSET" "$GAUGE_SIGMA" "$IMERG_SIGMA_FLOOR" \
        "$IMERG_REPRESENTATIVENESS" "$IMERG_R_MULTIPLIER" \
        "$GUIDANCE_GAMMA" "$GUIDANCE_SCALE" "$GUIDANCE_CLIP" "$HUBER_DELTA" \
        "$WITHHOLD" "$MIN_COVERAGE" "$HOLDOUT_NEIGHBOR_KM" \
        "$HOLDOUT_MAX_GAP_DEG" "$SEED" <<'PY'
import json
import math
import sys
import zarr

(
    store_path, report_path, target, checkpoint, imerg, start, end, members,
    steps, offset, gauge_sigma, imerg_sigma, imerg_rep, imerg_r_multiplier,
    gamma, scale, clip, huber, withhold, min_coverage, neighbor_km, max_gap, seed,
) = sys.argv[1:]
root = zarr.open_group(store_path, mode="r")
report = json.load(open(report_path))
assert root.attrs.get("complete", False)
assert root.attrs.get("diagnostic_complete", False)
assert root.attrs.get("archive_uses_likelihood_hard_decoder") is True
assert root.attrs.get("source_target_store") == target
assert root.attrs.get("source_checkpoint") == checkpoint
assert root.attrs.get("imerg_source") == imerg
assert root.attrs.get("condition_day_offset") == int(offset)
assert report["dates"] == [start, end]
assert report["members"] == int(members)
assert report["n_steps"] == int(steps)
assert report["seed"] == int(seed)

def close(left, right):
    assert math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-9)

station = report["stations"]
close(station["withhold_fraction"], withhold)
close(station["minimum_coverage"], min_coverage)
close(station["holdout_neighbor_km"], neighbor_km)
close(station["holdout_max_gap_deg"], max_gap)
observation = report["observation_model"]
close(observation["gauge_sigma_mm"], gauge_sigma)
close(observation["imerg_sigma_floor_mm"], imerg_sigma)
close(observation["imerg_representativeness_mm"], imerg_rep)
close(observation["imerg_r_multiplier"], imerg_r_multiplier)
close(observation["guidance"]["gamma"], gamma)
close(observation["guidance"]["scale"], scale)
close(observation["guidance"]["clip_norm"], clip)
close(observation["guidance"]["huber_delta"], huber)
PY
    then
        reuse_sample=1
        echo "Reusing completed, configuration-matched sample store: $SAMPLE_STORE"
    else
        echo "ERROR: existing sample store/report does not match this request" >&2
        echo "Set V4_TEST_OUT_ROOT to a new directory; files are never overwritten." >&2
        exit 1
    fi
elif [[
    "$RECOVER_INCOMPLETE" == "1" &&
    ! -e "$SAMPLE_STORE" &&
    -d "$SAMPLE_STORE.incomplete" &&
    ! -e "$SAMPLE_REPORT"
]]; then
    recover_sample=1
    echo "Recovering fully sampled incomplete archive without GPU resampling:"
    echo "  $SAMPLE_STORE.incomplete"
elif [[ "$RECOVER_INCOMPLETE" == "1" ]]; then
    echo "ERROR: recovery was requested, but the exact recoverable state is absent:" >&2
    echo "  expected $SAMPLE_STORE.incomplete" >&2
    echo "  expected no completed sample store or sampling report" >&2
    exit 1
elif [[ -e "$SAMPLE_STORE" || -e "$SAMPLE_STORE.incomplete" || -e "$SAMPLE_REPORT" ]]; then
    echo "ERROR: partial v4 diagnostic output exists under $OUT_ROOT" >&2
    echo "For the audited float32 roundoff failure, rerun with:" >&2
    echo "  V4_TEST_RECOVER_INCOMPLETE=1 bash slurm/submit_v4_subgrid_da_test.sh" >&2
    echo "Otherwise inspect it, then use a new V4_TEST_OUT_ROOT." >&2
    exit 1
fi

echo "Corrected CPC-V3-SG/v4 DA pilot"
echo "  dates:      $START through $END (conditioning offset $OFFSET day)"
echo "  checkpoint: $CHECKPOINT"
echo "  target:     $TARGET"
echo "  IMERG:      $IMERG (exact S04 footprint operator)"
echo "  sampling:   $MEMBERS members; $STEPS Heun steps; six matched-noise arms"
echo "  output:     $OUT_ROOT"
echo "  WARNING: short diagnostic only; not DA selection or confirmation"

if [[ "$reuse_sample" == "0" ]]; then
    recovery_args=()
    if [[ "$recover_sample" == "1" ]]; then
        recovery_args=(--recover-incomplete)
    fi
    srun --ntasks=1 "$PYTHON_BIN" -u scripts/60_v4_subgrid_da_test.py \
        --target-store "$TARGET" \
        --checkpoint "$CHECKPOINT" \
        --stations "$DAILY" \
        --imerg "$IMERG" \
        --start "$START" --end "$END" \
        --background-day-offset "$OFFSET" \
        --members "$MEMBERS" --n-steps "$STEPS" \
        --withhold "$WITHHOLD" --min-coverage "$MIN_COVERAGE" \
        --holdout-neighbor-km "$HOLDOUT_NEIGHBOR_KM" \
        --holdout-max-gap-deg "$HOLDOUT_MAX_GAP_DEG" \
        --gauge-sigma-mm "$GAUGE_SIGMA" \
        --imerg-sigma-floor-mm "$IMERG_SIGMA_FLOOR" \
        --imerg-representativeness-mm "$IMERG_REPRESENTATIVENESS" \
        --imerg-r-multiplier "$IMERG_R_MULTIPLIER" \
        --guidance-gamma "$GUIDANCE_GAMMA" \
        --guidance-scale "$GUIDANCE_SCALE" \
        --guidance-clip-norm "$GUIDANCE_CLIP" \
        --huber-delta "$HUBER_DELTA" \
        --seed "$SEED" \
        --out-store "$SAMPLE_STORE" \
        --report "$SAMPLE_REPORT" \
        "${recovery_args[@]}"
fi

"$PYTHON_BIN" -u scripts/58_evaluate_subgrid_prior.py \
    --target-store "$TARGET" \
    --sample-store "$SAMPLE_STORE" \
    --methods "$METHODS" \
    --out-dir "$GRIDDED_DIR"

"$PYTHON_BIN" -u scripts/61_evaluate_v4_subgrid_da_test.py \
    --target-store "$TARGET" \
    --sample-store "$SAMPLE_STORE" \
    --gridded-json "$GRIDDED_DIR/v3_subgrid_metrics.json" \
    --out-dir "$EVALUATION_DIR"

for result in \
    "$SAMPLE_STORE" \
    "$SAMPLE_REPORT" \
    "$EVALUATION_DIR/v4_da_test_metrics.json" \
    "$EVALUATION_DIR/v4_da_test_metrics.md" \
    "$EVALUATION_DIR/v4_da_test_matrix.png" \
    "$EVALUATION_DIR/v4_da_test_daily_maps.png" \
    "$EVALUATION_DIR/v4_da_test_subgrid_maps.png"; do
    [[ -e "$result" ]] || { echo "ERROR: expected output missing: $result" >&2; exit 1; }
done
echo "Corrected v4 diagnostic complete: $EVALUATION_DIR"
