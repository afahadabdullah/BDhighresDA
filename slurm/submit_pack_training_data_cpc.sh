#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PACK_CPC="${PACK_CPC:-data/raw/cpc}"
export PACK_OUT="${PACK_OUT:-data/processed/bd_wide_cpc.zarr}"
export PACK_QC_OUT="${PACK_QC_OUT:-data/processed/alignment_qc_cpc.json}"
export PACK_ALIGNMENT_CHANNEL="${PACK_ALIGNMENT_CHANNEL:-cpc_precip}"
exec "$SCRIPT_DIR/submit_pack_training_data.sh" "$@"
