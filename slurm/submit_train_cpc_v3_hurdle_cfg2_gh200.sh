#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG="${CONFIG:-configs/train_h100_cpc_v3_hurdle_cfg2.yaml}"
export TRAIN_PREFLIGHT_REPORT="${TRAIN_PREFLIGHT_REPORT:-data/processed/training_preflight_cpc_v3_hurdle.json}"
exec "$SCRIPT_DIR/submit_train_gh200.sh" "$@"
