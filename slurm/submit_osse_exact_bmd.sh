#!/usr/bin/env bash
# CHIRPS OSSE on the BMD geometry: perfect observations first, error later.
#
# Three headline arms, all with NO error added to either observation stream:
#   gauges_exact_bmd        pseudo-BMD gauges only
#   satellite_exact_bmd     pseudo-IMERG 0.1-degree footprints only
#   simultaneous_exact_bmd  both                                <- primary
#
# CHIRPS is the nature run and supplies both streams, so gauges and footprints
# are mutually consistent by construction and any disagreement between the arms
# is the assimilation's doing rather than the data's.
#
# Sensitivities with error added are arms 3-6 and are NOT submitted by default:
#   ARRAY=3-6%2 bash slurm/submit_osse_exact_bmd.sh
#
# Usage
#   bash slurm/submit_osse_exact_bmd.sh              # arms 0-2, perfect obs
#   ARRAY=1-2 bash slurm/submit_osse_exact_bmd.sh    # just the satellite arms
#   ARRAY=0-6%2 bash slurm/submit_osse_exact_bmd.sh  # everything

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

ROOT="${OSSE_BMD_ROOT:-data/processed/osse_bmd}"
ARRAY="${ARRAY:-0-2%2}"
PRIMARY="${OSSE_BMD_PRIMARY:-simultaneous_exact_bmd}"
CATALOG="${OSSE_BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"

if [[ ! -s "$CATALOG" ]]; then
    echo "ERROR: BMD station catalogue not found: $CATALOG" >&2
    echo "  scripts/10_osse.py defaults to data/bmd/Stations.csv, which does" >&2
    echo "  not exist here. Set OSSE_BMD_STATIONS to the catalogue with" >&2
    echo "  lat/lon columns." >&2
    exit 1
fi
echo "BMD catalogue: $CATALOG ($(($(wc -l < "$CATALOG") - 1)) rows)"
echo "Output root:   $ROOT"
echo "Array:         $ARRAY"
echo

submission="$(sbatch --parsable --array="$ARRAY" \
    --export="ALL,OSSE_BMD_ROOT=$ROOT,OSSE_BMD_STATIONS=$CATALOG" \
    slurm/osse_exact_bmd.sbatch)"
echo "submitted OSSE array: $submission"

# A failure here must not read as a failure of the array, which is already
# queued and holds all the GPU time.
if summary="$(sbatch --parsable --dependency="afterok:${submission}" \
        --export="ALL,OSSE_BMD_ROOT=$ROOT,OSSE_BMD_PRIMARY=$PRIMARY" \
        slurm/summarize_osse_bmd.sbatch 2>&1)"; then
    echo "submitted dependent figure job: $summary"
else
    echo
    echo "WARNING: the GPU array ($submission) SUBMITTED FINE; the dependent" >&2
    echo "figure job did not: $summary" >&2
    echo "When the array finishes, run it directly:" >&2
    echo "  OSSE_BMD_ROOT=$ROOT sbatch slurm/summarize_osse_bmd.sbatch" >&2
fi

cat <<EOF

 Read the three exact arms in this order.

 1. gauges_exact_bmd -- does the DA work at all with correct observations?
    Withheld CRPSS must be well above zero. If it is not, nothing else in the
    OSSE is interpretable.

 2. satellite_exact_bmd -- what do PERFECT 0.1-degree footprints buy?
    This is the upper bound on satellite assimilation for this system. The
    noisy version of this arm previously scored -68% CRPS; that was the
    observation-generation defect in docs/ablation_pseudo_satellite.tex, not
    a property of satellite information. If the exact arm is strongly POSITIVE
    the defect is confirmed and localised. If it is still negative, the problem
    is in the operator or the likelihood and not in how the error was drawn.

 3. simultaneous_exact_bmd -- do gauges and footprints combine?
    Compare against the better of the two single-source arms, not against the
    background. Perfect information from both should not be worse than either
    alone; if it is, the composite likelihood is mis-weighting them.

 Claim B is at its sharpest in arm 2. Its null is the truth's own 0.1-degree
 block mean, and that arm assimilates exactly that information -- so Claim B
 there measures purely what the PRIOR adds below the footprint, with no
 observation resolving it and no product circularity.

 Figures:
   OSSE_ROOT=$ROOT PRIMARY=$PRIMARY bash slurm/make_paper_figures.sh
EOF
