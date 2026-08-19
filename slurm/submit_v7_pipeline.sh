#!/usr/bin/env bash
set -euo pipefail

# V7: two independent stages, submitted together.
#
#   A  coarsen the CPCv2 archive to 0.1 deg ─▶ stats ─▶ train (scripts/train.py)
#   B  build the factor-2 subgrid archive    ─────────▶ train (scripts/57)
#
# Stage A is CPCv2's code path verbatim -- same packer output, same trainer, same
# hyperparameters -- one resolution up.  Stage B is V3-SG's allocation branch at
# factor 2.  They share no state at training time and compose only at inference,
# so a failure in one does not contaminate the other.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

MESO_CFG="${V7_MESO_CFG:-configs/train_v7_meso.yaml}"
ALLOC_CFG="${V7_ALLOC_CFG:-configs/train_v7_allocation.yaml}"
export V7_V2_ARCHIVE="${V7_V2_ARCHIVE:-data/processed/bd_wide_cpc.zarr}"
export V7_MESO_ARCHIVE="${V7_MESO_ARCHIVE:-data/processed/bd_wide_cpc_0p1.zarr}"
export V7_MESO_STATS="${V7_MESO_STATS:-data/processed/stats_v7_meso.json}"
export V7_PREP_OUT="${V7_PREP_OUT:-data/processed/v7/wide_v7.zarr}"
export V7_FACTOR="${V7_FACTOR:-2}"
export V7_COARSE_RES="${V7_COARSE_RES:-0.1}"
V7_RUN_TESTS="${V7_RUN_TESTS:-1}"

# The three CPU-only jobs -- tests, prepare-meso, prepare-alloc -- default to
# grace-cpuonly.  When that partition is short of capacity, point them at the GPU
# partition instead: none of them requests --gres, so they use a GPU node's Grace
# cores without ever reserving a GPU.  Slurm associations here reject
# multi-partition requests, so this is one name, not a list.
#
#   V7_CPU_PARTITION=grace bash slurm/submit_v7_pipeline.sh
cpu_partition=()
if [[ -n "${V7_CPU_PARTITION:-}" ]]; then
    cpu_partition=(--partition="$V7_CPU_PARTITION")
    echo "  CPU-only jobs -> partition $V7_CPU_PARTITION"
fi

for required in \
    scripts/71_v7_coarsen_pack_archive.py scripts/06_compute_stats.py scripts/train.py \
    scripts/56_build_chirps_subgrid_targets.py scripts/57_train_subgrid_oracle.py \
    slurm/v7_prepare_meso.sbatch "${V7_MESO_SBATCH:-slurm/train_h100.sbatch}" \
    slurm/v7_prepare.sbatch slurm/v7_train.sbatch \
    "$MESO_CFG" "$ALLOC_CFG"; do
    [[ -f "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done

# Stage A must read the archive stage A preparation writes, and must use the
# statistics measured on it -- reusing the 0.05-degree stats would normalise the
# model with the wrong moments and nothing downstream would notice.
meso_store="$(awk '$1 == "zarr:" {print $2; exit}' "$MESO_CFG")"
meso_stats="$(awk '$1 == "stats:" {print $2; exit}' "$MESO_CFG")"
[[ "$meso_store" == "$V7_MESO_ARCHIVE" ]] || {
    echo "ERROR: $MESO_CFG reads $meso_store, not $V7_MESO_ARCHIVE" >&2; exit 1; }
[[ "$meso_stats" == "$V7_MESO_STATS" ]] || {
    echo "ERROR: $MESO_CFG uses $meso_stats, not $V7_MESO_STATS" >&2; exit 1; }

# Stage A reuses the CPCv2 launcher, and the ONLY thing that points it at V7 is
# the CONFIG environment variable.  A launcher that pins its config instead would
# accept the submission, ignore CONFIG, and quietly retrain CPCv2 at 0.05 degrees
# for 150 epochs -- with V7's job name on it.  There is no later symptom, so the
# check has to happen here.
MESO_SBATCH="${V7_MESO_SBATCH:-slurm/train_h100.sbatch}"
grep -qE '\$\{?CONFIG' "$MESO_SBATCH" || {
    echo "ERROR: $MESO_SBATCH never reads \$CONFIG, so it would ignore $MESO_CFG" >&2
    echo "Point V7_MESO_SBATCH at the launcher CPCv2 actually used." >&2
    exit 1
}
# Every other V7 job asserts aarch64 and runs on the Grace partition.  A launcher
# named for H100s is worth naming out loud rather than discovering in the queue.
meso_partition="$(awk -F= '/^#SBATCH --partition/ {print $2; exit}' "$MESO_SBATCH")"
echo "  stage A launcher: $MESO_SBATCH  (partition ${meso_partition:-unset})"

alloc_store="$(awk '$1 == "zarr:" {print $2; exit}' "$ALLOC_CFG")"
alloc_factor="$(awk '$1 == "factor:" {print $2; exit}' "$ALLOC_CFG")"
[[ "$alloc_store" == "$V7_PREP_OUT" ]] || {
    echo "ERROR: $ALLOC_CFG reads $alloc_store, not $V7_PREP_OUT" >&2; exit 1; }
[[ "$alloc_factor" == "$V7_FACTOR" ]] || {
    echo "ERROR: $ALLOC_CFG uses factor $alloc_factor, not $V7_FACTOR" >&2; exit 1; }

echo "Submitting V7"
echo "  stage A: CPCv2 path at ${V7_COARSE_RES} deg   $V7_V2_ARCHIVE -> $V7_MESO_ARCHIVE"
echo "  stage B: allocation, factor ${V7_FACTOR}       $V7_PREP_OUT"

dependency=()
if [[ "$V7_RUN_TESTS" == "1" ]]; then
    tests="$(sbatch --parsable "${cpu_partition[@]}" --export=ALL "$@" slurm/v3_subgrid_tests.sbatch)"
    dependency=(--dependency="afterok:${tests}" --kill-on-invalid-dep=yes)
    echo "submitted tests:        $tests"
fi

prep_a="$(sbatch --parsable "${dependency[@]}" "${cpu_partition[@]}" --export=ALL "$@" slurm/v7_prepare_meso.sbatch)"
echo "submitted prepare-meso: $prep_a"

meso="$(CONFIG="$MESO_CFG" \
    TRAIN_PREFLIGHT_REPORT="data/processed/training_preflight_v7_meso.json" \
    sbatch --parsable --job-name=bdhires-v7-meso \
        --dependency="afterok:${prep_a}" --kill-on-invalid-dep=yes \
        --export=ALL "$@" "$MESO_SBATCH")"
echo "submitted meso:         $meso (afterok:$prep_a)"

prep_b="$(sbatch --parsable "${dependency[@]}" "${cpu_partition[@]}" --export=ALL "$@" slurm/v7_prepare.sbatch)"
echo "submitted prepare-alloc: $prep_b"

alloc="$(V7_CONFIG="$ALLOC_CFG" V7_EXPECTED_STAGE=allocation \
    sbatch --parsable --job-name=bdhires-v7-allocation \
        --dependency="afterok:${prep_b}" --kill-on-invalid-dep=yes \
        --export=ALL "$@" slurm/v7_train.sbatch)"
echo "submitted allocation:    $alloc (afterok:$prep_b)"

echo "monitor: squeue -u $USER"
echo "final:   runs/v7/meso/best.pt and runs/v7/allocation/best.pt"
