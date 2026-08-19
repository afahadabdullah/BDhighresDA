#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

export V4_TEST_START="${V4_TEST_START:-2022-05-01}"
export V4_TEST_END="${V4_TEST_END:-2022-05-05}"
export V4_TEST_MEMBERS="${V4_TEST_MEMBERS:-4}"
export V4_TEST_STEPS="${V4_TEST_STEPS:-25}"
export V4_TEST_BACKGROUND_OFFSET="${V4_TEST_BACKGROUND_OFFSET:--1}"
export V4_TEST_CHECKPOINT="${V4_TEST_CHECKPOINT:-runs/prior_h100_cpc_v3_subgrid_v4/joint/best.pt}"
export V4_TEST_TARGET="${V4_TEST_TARGET:-data/processed/cpc_v3_subgrid/wide_cpc_v4.zarr}"
export V4_TEST_IMERG="${V4_TEST_IMERG:-data/processed/imerg_prepared_ing2022/imerg_0p4deg_20220501_20220510.nc}"
export V4_TEST_OUT_ROOT="${V4_TEST_OUT_ROOT:-data/processed/v4_da_test/may2022_5day}"
export V4_TEST_RECOVER_INCOMPLETE="${V4_TEST_RECOVER_INCOMPLETE:-0}"
export BMD_DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
export BMD_STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"

[[ "$V4_TEST_RECOVER_INCOMPLETE" == "0" || "$V4_TEST_RECOVER_INCOMPLETE" == "1" ]] || {
    echo "ERROR: V4_TEST_RECOVER_INCOMPLETE must be 0 or 1" >&2
    exit 1
}

for required in \
    "$V4_TEST_TARGET" \
    "$V4_TEST_IMERG" \
    "$BMD_DATA_DIR" \
    "$BMD_STATIONS" \
    scripts/60_v4_subgrid_da_test.py \
    scripts/61_evaluate_v4_subgrid_da_test.py \
    slurm/v4_subgrid_da_test.sbatch; do
    [[ -e "$required" ]] || {
        echo "ERROR: missing required v4 diagnostic input: $required" >&2
        exit 1
    }
done

dependency=()
if [[ -n "${V4_JOINT_JOB_ID:-}" ]]; then
    [[ "$V4_JOINT_JOB_ID" =~ ^[0-9]+$ ]] || {
        echo "ERROR: V4_JOINT_JOB_ID must be a numeric Slurm job ID" >&2
        exit 1
    }
    # Without kill-on-invalid-dep, a failed training job leaves this one parked
    # in the queue indefinitely rather than failing where it can be noticed.
    dependency=(
        --dependency="afterok:${V4_JOINT_JOB_ID}"
        --kill-on-invalid-dep=yes
    )
elif [[ ! -f "$V4_TEST_CHECKPOINT" ]]; then
    echo "ERROR: joint best checkpoint is not present: $V4_TEST_CHECKPOINT" >&2
    echo "Wait for training, or set V4_JOINT_JOB_ID to submit with afterok dependency." >&2
    exit 1
fi

echo "Submitting corrected v4 five-day DA diagnostic"
echo "  dates:      $V4_TEST_START through $V4_TEST_END"
echo "  checkpoint: $V4_TEST_CHECKPOINT"
echo "  sampling:   $V4_TEST_MEMBERS members; $V4_TEST_STEPS Heun steps"
echo "  arms:       background, folded gauges, IMERG S04, folded simultaneous,"
echo "              all-gauge maps, all-gauge simultaneous maps"
echo "  output:     $V4_TEST_OUT_ROOT"
if [[ "$V4_TEST_RECOVER_INCOMPLETE" == "1" ]]; then
    echo "  recovery:   reuse fully sampled .incomplete states; no resampling"
fi
if [[ -n "${V4_JOINT_JOB_ID:-}" ]]; then
    echo "  dependency: afterok:$V4_JOINT_JOB_ID"
fi

result="$(sbatch --parsable "${dependency[@]}" --export=ALL "$@" \
    slurm/v4_subgrid_da_test.sbatch)"
job="${result%%;*}"
echo "submitted v4 DA diagnostic: $job"
echo "monitor: tail -f logs/bdhires-v4-da-test-${job}.out"
echo "matrix:  $V4_TEST_OUT_ROOT/evaluation/v4_da_test_matrix.png"
echo "maps:    $V4_TEST_OUT_ROOT/evaluation/v4_da_test_daily_maps.png"
echo "subgrid: $V4_TEST_OUT_ROOT/evaluation/v4_da_test_subgrid_maps.png"
echo "table:   $V4_TEST_OUT_ROOT/evaluation/v4_da_test_metrics.md"
