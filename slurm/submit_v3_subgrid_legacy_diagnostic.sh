#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

export LEGACY_V3_START="${LEGACY_V3_START:-2022-05-01}"
export LEGACY_V3_END="${LEGACY_V3_END:-2022-05-05}"
export LEGACY_V3_MEMBERS="${LEGACY_V3_MEMBERS:-4}"
export LEGACY_V3_STEPS="${LEGACY_V3_STEPS:-25}"
export LEGACY_V3_BACKGROUND_OFFSET="${LEGACY_V3_BACKGROUND_OFFSET:--1}"
export LEGACY_V3_GAUGE_SIGMA_MM="${LEGACY_V3_GAUGE_SIGMA_MM:-3.0}"
export LEGACY_V3_GUIDANCE_GAMMA="${LEGACY_V3_GUIDANCE_GAMMA:-1.0}"
export LEGACY_V3_GUIDANCE_SCALE="${LEGACY_V3_GUIDANCE_SCALE:-1.0}"
export LEGACY_V3_GUIDANCE_CLIP_NORM="${LEGACY_V3_GUIDANCE_CLIP_NORM:-100.0}"
export LEGACY_V3_HUBER_DELTA="${LEGACY_V3_HUBER_DELTA:-3.0}"
export LEGACY_V3_CHECKPOINT="${LEGACY_V3_CHECKPOINT:-runs/prior_h100_cpc_v3_subgrid/joint/best.pt}"
export LEGACY_V3_TARGET="${LEGACY_V3_TARGET:-data/processed/cpc_v3_subgrid/wide_cpc.zarr}"
export LEGACY_V3_OUT_DIR="${LEGACY_V3_OUT_DIR:-data/processed/v3_legacy_diagnostic/may2022_5day}"
export BMD_DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
export BMD_STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"

for required in \
    "$LEGACY_V3_CHECKPOINT" \
    "$LEGACY_V3_TARGET" \
    "$BMD_DATA_DIR" \
    "$BMD_STATIONS" \
    scripts/59_legacy_v3_subgrid_diagnostic.py \
    slurm/v3_subgrid_legacy_diagnostic.sbatch; do
    [[ -e "$required" ]] || {
        echo "ERROR: missing required legacy diagnostic input: $required" >&2
        exit 1
    }
done

if [[ -e "$LEGACY_V3_OUT_DIR/legacy_v3_may2022_5day.zarr" ]]; then
    echo "ERROR: diagnostic output already exists:" >&2
    echo "  $LEGACY_V3_OUT_DIR/legacy_v3_may2022_5day.zarr" >&2
    echo "Set LEGACY_V3_OUT_DIR to a new directory for another run." >&2
    exit 1
fi

echo "Submitting legacy pre-v4 V3-SG five-day diagnostic"
echo "  dates:      $LEGACY_V3_START through $LEGACY_V3_END"
echo "  checkpoint: $LEGACY_V3_CHECKPOINT"
echo "  target:     $LEGACY_V3_TARGET"
echo "  sampling:   $LEGACY_V3_MEMBERS members; $LEGACY_V3_STEPS Heun steps"
echo "  gauge DA:   sigma=$LEGACY_V3_GAUGE_SIGMA_MM mm; gamma=$LEGACY_V3_GUIDANCE_GAMMA"
echo "  output:     $LEGACY_V3_OUT_DIR"
echo "  WARNING: diagnostic only; corrected v4 remains the production experiment"

result="$(sbatch --parsable --export=ALL "$@" slurm/v3_subgrid_legacy_diagnostic.sbatch)"
job="${result%%;*}"
echo "submitted legacy diagnostic: $job"
echo "monitor: tail -f logs/bdhires-v3-legacy-diag-${job}.out"
echo "figure:  $LEGACY_V3_OUT_DIR/legacy_v3_may2022_5day.png"
echo "subgrid: $LEGACY_V3_OUT_DIR/legacy_v3_may2022_5day_subgrid.png"
echo "report:  $LEGACY_V3_OUT_DIR/legacy_v3_may2022_5day.json"
