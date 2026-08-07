#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PREFLIGHT_CONFIG="${PREFLIGHT_CONFIG:-configs/train_h100_cpc_v3.yaml}"
# Shared with every other config on bd_wide_cpc.zarr + stats_cpc.json.
# scripts/06_plot_normalization.py reads only the zarr and the stats, so a
# per-ablation report would be a byte-identical copy -- and naming three files
# that nothing generates is what made the first v3 preflight fail.
export PREFLIGHT_NORM_REPORT="${PREFLIGHT_NORM_REPORT:-data/processed/normalization_diagnostics_cpc.json}"
export PREFLIGHT_OUT="${PREFLIGHT_OUT:-data/processed/training_preflight_cpc_v3.json}"
exec "$SCRIPT_DIR/submit_preflight_training_gh200.sh" "$@"
