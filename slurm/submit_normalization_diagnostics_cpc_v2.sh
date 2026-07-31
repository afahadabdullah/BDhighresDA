#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NORM_ZARR="${NORM_ZARR:-data/processed/bd_wide_cpc.zarr}"
export NORM_STATS="${NORM_STATS:-data/processed/stats_cpc_v2.json}"
export NORM_FIGURE="${NORM_FIGURE:-data/processed/normalization_diagnostics_cpc_v2.png}"
export NORM_REPORT="${NORM_REPORT:-data/processed/normalization_diagnostics_cpc_v2.json}"
exec "$SCRIPT_DIR/submit_normalization_diagnostics.sh" "$@"
