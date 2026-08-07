#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PREFLIGHT_CONFIG="${PREFLIGHT_CONFIG:-configs/train_h100_cpc_v3.yaml}"
export PREFLIGHT_NORM_REPORT="${PREFLIGHT_NORM_REPORT:-data/processed/normalization_diagnostics_cpc_v3.json}"
export PREFLIGHT_OUT="${PREFLIGHT_OUT:-data/processed/training_preflight_cpc_v3.json}"
exec "$SCRIPT_DIR/submit_preflight_training_gh200.sh" "$@"
