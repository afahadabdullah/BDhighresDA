#!/usr/bin/env bash
set -euo pipefail

# Five-day screening sweep over simultaneous-DA methods, May 2024.
#
#   bash slurm/submit_simultaneous_sweep_may2024.sh            # core group
#   SWEEP_GROUP=weighting bash slurm/submit_simultaneous_sweep_may2024.sh
#   SWEEP_GROUP=all SWEEP_MEMBERS=8 bash slurm/submit_simultaneous_sweep_may2024.sh
#
# May 2024 is chosen because it is outside the checkpoint's training span
# (train 1981-2018, val 2019-2020, test 2021-2025) and because the 2024 arm of
# the pooled evaluation covers May-June only, so this window is directly
# comparable to an existing result.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"

export SWEEP_START="${SWEEP_START:-2024-05-01}"
export SWEEP_END="${SWEEP_END:-2024-05-05}"
export SWEEP_GROUP="${SWEEP_GROUP:-core}"
export SWEEP_MEMBERS="${SWEEP_MEMBERS:-16}"
export SWEEP_FOLD="${SWEEP_FOLD:-0}"
export SWEEP_FOLDS="${SWEEP_FOLDS:-5}"
export SWEEP_ROOT="${SWEEP_ROOT:-data/processed/method_sweep}"
export SWEEP_BASELINE="${SWEEP_BASELINE:-gauges_only}"

export BMD_CKPT="${BMD_CKPT:-runs/prior_h100_cpc/best.pt}"
export BMD_CONFIG="${BMD_CONFIG:-configs/da.yaml}"
export BACKGROUND_DAY_OFFSET="${BACKGROUND_DAY_OFFSET:--1}"

# Leave-one-year-out quantile map. These are the prepared IMERG files the
# multi-year evaluation already produced; 2024 is held out when the map is
# applied to a 2024 window, so no CHIRPS from the evaluation year enters it.
export SWEEP_IMERG_QM="${SWEEP_IMERG_QM:-data/processed/imerg_qm_loyo.npz}"
export SWEEP_QM_SOURCES="${SWEEP_QM_SOURCES:-\
data/processed/bmd_imerg_eval_2021_may_sep/imerg_aligned_20210501_20210930.nc \
data/processed/bmd_imerg_eval_2022_may_sep/imerg_aligned_20220501_20220930.nc \
data/processed/bmd_imerg_eval_2023_may_sep/imerg_aligned_20230501_20230930.nc \
data/processed/bmd_imerg_eval_2024_may_jun/imerg_aligned_20240501_20240630.nc}"
export SWEEP_ZARR="${SWEEP_ZARR:-data/processed/bd_wide_cpc.zarr}"
export SWEEP_STATS="${SWEEP_STATS:-data/processed/stats_cpc.json}"

mkdir -p logs "$SWEEP_ROOT"
exec sbatch "$@" slurm/simultaneous_method_sweep.sbatch
