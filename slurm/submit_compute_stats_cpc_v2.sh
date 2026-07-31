#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export STATS_ZARR="${STATS_ZARR:-data/processed/bd_wide_cpc.zarr}"
export STATS_ALIGNMENT_REPORT="${STATS_ALIGNMENT_REPORT:-data/processed/alignment_qc_cpc.json}"
export STATS_OUT="${STATS_OUT:-data/processed/stats_cpc_v2.json}"
export STATS_TRANSFORM="${STATS_TRANSFORM:-sqrt}"
export STATS_CPC_PRECIP_TRANSFORM="${STATS_CPC_PRECIP_TRANSFORM:-sqrt}"
export STATS_DAILY_WETNESS="${STATS_DAILY_WETNESS:-1}"
export STATS_RESIDUAL="${STATS_RESIDUAL:-1}"
export STATS_RESIDUAL_BASE="${STATS_RESIDUAL_BASE:-cpc_precip}"
export STATS_RESIDUAL_BASE_INDEX="${STATS_RESIDUAL_BASE_INDEX:-6}"
exec "$SCRIPT_DIR/submit_compute_stats.sh" "$@"
