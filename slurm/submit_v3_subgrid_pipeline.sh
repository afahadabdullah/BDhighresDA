#!/usr/bin/env bash
set -euo pipefail

# Submit archive preparation, parallel branch training and dependent joint
# training. Reruns reuse a completed archive and resume exact-config checkpoints.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

for required in \
    scripts/56_build_chirps_subgrid_targets.py \
    scripts/57_train_subgrid_oracle.py \
    scripts/03_build_static.py \
    slurm/v3_subgrid_tests.sbatch \
    configs/train_h100_cpc_v3_subgrid_coarse.yaml \
    configs/train_h100_cpc_v3_subgrid_allocation.yaml \
    configs/train_h100_cpc_v3_subgrid_joint.yaml; do
    [[ -f "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done

export V3_CHIRPS_GLOB="${V3_CHIRPS_GLOB:-data/raw/chirps/chirps_wide_*.nc}"
export V3_CPC_GLOB="${V3_CPC_GLOB:-data/raw/cpc/precip.*.nc}"
export V3_ERA5_GLOB="${V3_ERA5_GLOB:-data/raw/era5/era5_daily_*.nc}"
export V3_STATIC="${V3_STATIC:-data/static/static_wide_cpc.nc}"
export V3_DEM="${V3_DEM:-data/raw/dem/copernicus_glo90_wide.nc}"
export V3_STATIC_CHIRPS="${V3_STATIC_CHIRPS:-}"
export V3_PREP_OUT="${V3_PREP_OUT:-data/processed/cpc_v3_subgrid/wide_cpc_v4.zarr}"
export V3_PREP_CHUNK_DAYS="${V3_PREP_CHUNK_DAYS:-32}"
export V3_PREP_OVERWRITE="${V3_PREP_OVERWRITE:-0}"
export V3_START="${V3_START:-1981-01-01}"
export V3_END="${V3_END:-2024-12-31}"
export V3_RUN_TESTS="${V3_RUN_TESTS:-1}"
[[ "$V3_RUN_TESTS" == "0" || "$V3_RUN_TESTS" == "1" ]] || {
    echo "ERROR: V3_RUN_TESTS must be 0 or 1" >&2
    exit 1
}

# The three frozen configs all point here. An output override without matching
# config overrides would prepare one archive and train against another.
EXPECTED_TARGET="data/processed/cpc_v3_subgrid/wide_cpc_v4.zarr"
if [[ "$V3_PREP_OUT" != "$EXPECTED_TARGET" ]]; then
    echo "ERROR: V3_PREP_OUT=$V3_PREP_OUT differs from the frozen config target" >&2
    echo "Update all three data.zarr entries together before using a custom path." >&2
    exit 1
fi

# Catch missing raw prerequisites on the login node so a four-job dependency
# chain is not submitted only to fail immediately in the preparation job.
if [[ ! -e "$V3_PREP_OUT" || "$V3_PREP_OVERWRITE" == "1" ]]; then
    for label_and_pattern in \
        "CHIRPS|$V3_CHIRPS_GLOB" \
        "CPC|$V3_CPC_GLOB" \
        "ERA5|$V3_ERA5_GLOB"; do
        label="${label_and_pattern%%|*}"
        pattern="${label_and_pattern#*|}"
        if ! compgen -G "$pattern" >/dev/null; then
            echo "ERROR: $label input pattern matched no files: $pattern" >&2
            echo "No Slurm jobs were submitted." >&2
            exit 1
        fi
    done
    if [[ ! -f "$V3_STATIC" ]]; then
        [[ -f "$V3_DEM" ]] || {
            echo "ERROR: missing source DEM needed to build static fields: $V3_DEM" >&2
            echo "No Slurm jobs were submitted." >&2
            exit 1
        }
        if [[ -n "$V3_STATIC_CHIRPS" && ! -f "$V3_STATIC_CHIRPS" ]]; then
            echo "ERROR: missing V3_STATIC_CHIRPS: $V3_STATIC_CHIRPS" >&2
            echo "No Slurm jobs were submitted." >&2
            exit 1
        fi
    fi
fi

echo "Submitting V3-SG training pipeline"
echo "  prepare:    CHIRPS/CPC/ERA5 ($V3_START through $V3_END) -> $V3_PREP_OUT"
echo "  static:     reuse $V3_STATIC, or build it from existing CHIRPS + $V3_DEM"
echo "  branches:   coarse and allocation run in parallel"
echo "  joint:      starts only after both branch jobs succeed"
echo "  checkpoint: interrupted stages resume only with an identical config"

prepare_dependency=()
test_job=""
if [[ "$V3_RUN_TESTS" == "1" ]]; then
    test_result="$(sbatch --parsable --export=ALL "$@" \
        slurm/v3_subgrid_tests.sbatch)"
    test_job="${test_result%%;*}"
    prepare_dependency=(--dependency="afterok:${test_job}")
fi

prepare_result="$(sbatch --parsable "${prepare_dependency[@]}" --export=ALL "$@" \
    slurm/v3_subgrid_prepare.sbatch)"
prepare_job="${prepare_result%%;*}"

export V3_CONFIG="configs/train_h100_cpc_v3_subgrid_coarse.yaml"
export V3_EXPECTED_STAGE="coarse"
coarse_result="$(sbatch --parsable --job-name=bdhires-v3-coarse \
    --dependency="afterok:${prepare_job}" --export=ALL "$@" \
    slurm/v3_subgrid_train.sbatch)"
coarse_job="${coarse_result%%;*}"

export V3_CONFIG="configs/train_h100_cpc_v3_subgrid_allocation.yaml"
export V3_EXPECTED_STAGE="allocation"
allocation_result="$(sbatch --parsable --job-name=bdhires-v3-allocation \
    --dependency="afterok:${prepare_job}" --export=ALL "$@" \
    slurm/v3_subgrid_train.sbatch)"
allocation_job="${allocation_result%%;*}"

export V3_CONFIG="configs/train_h100_cpc_v3_subgrid_joint.yaml"
export V3_EXPECTED_STAGE="joint"
joint_result="$(sbatch --parsable --job-name=bdhires-v3-joint \
    --dependency="afterok:${coarse_job}:${allocation_job}" --export=ALL "$@" \
    slurm/v3_subgrid_train.sbatch)"
joint_job="${joint_result%%;*}"

if [[ -n "$test_job" ]]; then
    echo "submitted tests:       $test_job"
fi
echo "submitted preparation: $prepare_job${test_job:+ (afterok:$test_job)}"
echo "submitted coarse:      $coarse_job (afterok:$prepare_job)"
echo "submitted allocation:  $allocation_job (afterok:$prepare_job)"
echo "submitted joint:       $joint_job (afterok:$coarse_job:$allocation_job)"
echo "monitor: squeue -u $USER"
echo "logs:    logs/bdhires-v3-{tests,prepare,coarse,allocation,joint}-*.out"
echo "final:   runs/prior_h100_cpc_v3_subgrid_v4/joint/best.pt"
