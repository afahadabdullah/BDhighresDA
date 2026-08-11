#!/usr/bin/env bash
set -euo pipefail

# Non-overlapping second-stage arms for the CPC-v2 gauges-only tournament.
# The background is the sole repeated arm and supplies the paired reference.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export V2_SWEEP_GROUP="v2_gauges_refine"
export V2_SWEEP_LABEL="${V2_SWEEP_LABEL:-ing2022_refine}"
export V2_SWEEP_CURRENT="background"

exec bash "$SCRIPT_DIR/submit_v2_gauge_method_sweep.sh" "$@"
