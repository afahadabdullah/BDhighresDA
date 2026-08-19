#!/usr/bin/env bash
set -euo pipefail
# Stage A alone, through the CPCv2 launcher.  Mirrors
# submit_train_cpc_v2_gh200.sh exactly; only the config and preflight name differ.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG="${CONFIG:-configs/train_v7_meso.yaml}"
export TRAIN_PREFLIGHT_REPORT="${TRAIN_PREFLIGHT_REPORT:-data/processed/training_preflight_v7_meso.json}"
exec "$SCRIPT_DIR/submit_train_gh200.sh" "$@"
