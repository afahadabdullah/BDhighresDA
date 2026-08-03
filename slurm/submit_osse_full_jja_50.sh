#!/usr/bin/env bash
# Full controlled gauge-density ablation: 50 spread CHIRPS pseudo-gauges,
# 40 assimilated and 10 withheld, alongside matched exact 0.1-degree arms.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OSSE_FULL_ROOT="${OSSE_FULL_ROOT:-data/processed/osse_full_jja_50_2021_2024}"
export OSSE_FULL_NETWORK=50
export OSSE_FULL_NETWORK_TAG=50

exec bash "$SCRIPT_DIR/submit_osse_full_jja.sh" "$@"
