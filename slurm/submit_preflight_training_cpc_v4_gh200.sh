#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PREFLIGHT_CONFIG="${PREFLIGHT_CONFIG:-configs/train_h100_cpc_v4.yaml}"
# Shared: 06_plot_normalization reads only the zarr and the stats, and v4 uses
# the same pair as every other cpc config.
export PREFLIGHT_NORM_REPORT="${PREFLIGHT_NORM_REPORT:-data/processed/normalization_diagnostics_cpc.json}"
export PREFLIGHT_OUT="${PREFLIGHT_OUT:-data/processed/training_preflight_cpc_v4.json}"
exec "$SCRIPT_DIR/submit_preflight_training_gh200.sh" "$@"
