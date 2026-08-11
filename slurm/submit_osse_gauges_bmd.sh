#!/usr/bin/env bash
# Gauges-only OSSE on the real BMD geometry, then the paper figures.
#
# WHY THE SATELLITE IS OUT
# In the CHIRPS OSSE the pseudo-satellite arm scored -68% CRPS and -146% RMSE
# against its own background, and the simultaneous arm -39% / -91%. Those are
# not "the satellite is less informative" numbers; an observation that cannot
# help should converge on zero improvement, not drive the analysis far below
# the prior it started from.
#
# The metric pattern says where to look. Subgrid correlation gain went UP with
# pseudo-IMERG (+0.038, against +0.018 for gauges alone) and was best in the
# simultaneous arm (+0.047), while coverage90 also improved. Structure right,
# magnitude catastrophically wrong is a UNITS or TRANSFORM signature, not a
# noise one -- noise degrades correlation and RMSE together. The likely cause
# is that T(mean(x)) != mean(T(x)) for the precipitation transform: the
# operator is PhysicalBlockAverageObsOperator, which averages in physical space
# and then transforms, so pseudo-observations built by averaging TRANSFORMED
# CHIRPS would carry an intensity-dependent offset on every footprint.
#
# That is a real bug to find, not a result to report, and it contaminates every
# arm it touches. Until it is found the satellite is excluded, and the OSSE
# runs gauges-only so the DA method can be evaluated on something trustworthy.
#
# Usage
#   bash slurm/submit_osse_gauges_bmd.sh
#   OSSE_GAUGE_DAYS=96 bash slurm/submit_osse_gauges_bmd.sh
#   ARRAY=0-1 bash slurm/submit_osse_gauges_bmd.sh    # skip the tuning arm

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

ROOT="${OSSE_GAUGE_ROOT:-data/processed/osse_gauges}"
ARRAY="${ARRAY:-0-2%2}"
CATALOG="${OSSE_BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"

if [[ ! -s "$CATALOG" ]]; then
    echo "ERROR: BMD station catalogue not found: $CATALOG" >&2
    echo "  scripts/10_osse.py defaults to data/bmd/Stations.csv, which does" >&2
    echo "  not exist here. Set OSSE_BMD_STATIONS to the catalogue with" >&2
    echo "  lat/lon columns (the per-station archive is under" >&2
    echo "  data/stations/data_2020_2025/)." >&2
    exit 1
fi
echo "BMD catalogue: $CATALOG ($(($(wc -l < "$CATALOG") - 1)) rows)"
echo "Output root:   $ROOT"
echo

submission="$(sbatch --parsable --array="$ARRAY" \
    --export="ALL,OSSE_GAUGE_ROOT=$ROOT,OSSE_BMD_STATIONS=$CATALOG" \
    slurm/osse_gauges_bmd.sbatch)"
echo "submitted gauges-only OSSE array: $submission"

summary="$(sbatch --parsable --dependency="afterok:${submission}" \
    --export="ALL,OSSE_GAUGE_ROOT=$ROOT" \
    slurm/summarize_osse_gauges_bmd.sbatch)"
echo "submitted dependent figure job:   $summary"

cat <<EOF

 Arms (no satellite in any of them):
   gauges_exact_bmd      exact pseudo-observations -- the mechanistic bound
   gauges_realistic_bmd  observation error from configs/da.yaml
   gauges_tune_bmd       gamma x sigma_obs x prior_temperature, 60 combinations

 Everything is scored on WITHHELD pseudo-gauges at real BMD coordinates:
 8 withheld of ~40, matching the real-data folds so the two experiments can be
 read against each other.

 The tuning arm is the one to look at first. On real data, using the MEASURED
 representativeness (0.410) was among the WORST settings, and the proposed
 explanation was the prior's +8.37 mm/day wet bias rather than a bad
 measurement. The grid here includes sigma_obs = 0.40 against a KNOWN truth:

   sigma_obs = 0.40 near-optimal here  -> the explanation holds; the
                                          measurement is fine and the prior
                                          is what is wrong
   sigma_obs = 0.40 poor here too      -> the explanation fails and the
                                          representativeness estimate needs
                                          revisiting

 When the array finishes:
   OSSE_ROOT=$ROOT PRIMARY=gauges_realistic_bmd \\
       bash slurm/make_paper_figures.sh
EOF
