#!/usr/bin/env bash
set -euo pipefail

# Three-arm frozen-checkpoint June confirmation. This deliberately ignores the
# latest/latest tournament pair and returns to the checkpoint pair that produced
# the original May results.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"

export V7_JUNE_ROOT="${V7_JUNE_ROOT:-data/processed/v7_june2023_three_arm_frozen}"
export V7_JUNE_ARM_SET="three_arm"
export V7_JUNE_COMPARISON_LABEL="june2023_v7_three_arm_frozen_vs_cpcv2"
export V7_JUNE_MESO_CKPT="${V7_JUNE_MESO_CKPT:-data/processed/v7_osse/20260820_1356/checkpoints/meso_frozen.pt}"
export V7_JUNE_ALLOC_CKPT="${V7_JUNE_ALLOC_CKPT:-data/processed/v7_osse/20260820_1356/checkpoints/allocation_frozen.pt}"

exec bash slurm/submit_v7_june2023_r81.sh "$@"
