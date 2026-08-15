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
    configs/train_h100_cpc_v3_subgrid_coarse.yaml \
    configs/train_h100_cpc_v3_subgrid_allocation.yaml \
    configs/train_h100_cpc_v3_subgrid_joint.yaml; do
    [[ -f "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done

export V3_CHIRPS_GLOB="${V3_CHIRPS_GLOB:-data/raw/chirps_v3sg/chirps_wide_cpc_*.nc}"
export V3_CPC_GLOB="${V3_CPC_GLOB:-data/raw/cpc/precip.*.nc}"
export V3_ERA5_GLOB="${V3_ERA5_GLOB:-data/raw/era5_v3sg/era5_daily_*.nc}"
export V3_STATIC="${V3_STATIC:-data/static/static_wide_cpc.nc}"
export V3_PREP_OUT="${V3_PREP_OUT:-data/processed/cpc_v3_subgrid/wide_cpc.zarr}"
export V3_PREP_CHUNK_DAYS="${V3_PREP_CHUNK_DAYS:-32}"
export V3_PREP_OVERWRITE="${V3_PREP_OVERWRITE:-0}"

# The three frozen configs all point here. An output override without matching
# config overrides would prepare one archive and train against another.
EXPECTED_TARGET="data/processed/cpc_v3_subgrid/wide_cpc.zarr"
if [[ "$V3_PREP_OUT" != "$EXPECTED_TARGET" ]]; then
    echo "ERROR: V3_PREP_OUT=$V3_PREP_OUT differs from the frozen config target" >&2
    echo "Update all three data.zarr entries together before using a custom path." >&2
    exit 1
fi

echo "Submitting V3-SG training pipeline"
echo "  prepare:    CHIRPS/CPC/ERA5 -> $V3_PREP_OUT"
echo "  branches:   coarse and allocation run in parallel"
echo "  joint:      starts only after both branch jobs succeed"
echo "  checkpoint: interrupted stages resume only with an identical config"

prepare_result="$(sbatch --parsable --export=ALL "$@" \
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

echo "submitted preparation: $prepare_job"
echo "submitted coarse:      $coarse_job (afterok:$prepare_job)"
echo "submitted allocation:  $allocation_job (afterok:$prepare_job)"
echo "submitted joint:       $joint_job (afterok:$coarse_job:$allocation_job)"
echo "monitor: squeue -u $USER"
echo "logs:    logs/bdhires-v3-{prepare,coarse,allocation,joint}-*.out"
echo "final:   runs/prior_h100_cpc_v3_subgrid/joint/best.pt"
