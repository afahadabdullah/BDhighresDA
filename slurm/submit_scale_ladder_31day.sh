#!/usr/bin/env bash
# Scale-ladder confirmation: the same three footprints over a month, not ten days.
#
# Results being followed up: docs/ingestion_results.tex
#
# WHY THIS RUN EXISTS
# The eleven-arm screen over 2022-05-01..10 found no significant point-skill
# difference between any arm and gauges-only, but a clean monotone STRUCTURAL
# trend across the scale ladder: as the IMERG footprint coarsens from 0.1 to
# 0.8 degrees, pattern correlation rises (0.50 -> 0.59 -> 0.67), wet-area
# fraction falls toward plausibility (0.922 -> 0.875 -> 0.820) and bias falls
# (+1.84 -> +2.16 -> +1.19). The background's own pattern correlation is 0.60,
# so assimilating at 0.1 deg makes the field WORSE than not assimilating at
# all, and only 0.8 deg improves on it.
#
# WHY 31 DAYS
# The point test was underpowered, not null. S04's interval half-width was
# 0.33 mm/day against a point estimate of 0.192, so excluding zero needs about
# (0.33/0.192)^2 = 3.0x the sample: ~1130 withheld station-days against the 380
# we had. Thirty-one days x five folds x eight withheld is ~1240. The number
# comes from the measurement, not from rounding up.
#
# WHY ONLY THREE ARMS
# The strength axis produced ten intervals all containing zero and a monotone
# but insignificant ordering; there is nothing there to confirm. S1 is
# established as harmful on two independent windows and is not worth more GPU
# time. RAW is kept as the foot of the ladder and the baseline.
#
# CAVEAT, RECORDED SO IT IS NOT FORGOTTEN
# May 2022 in full is a strict SUPERSET of the ten days already used. This buys
# statistical power, NOT an independent replication -- the earlier sample is
# nested inside it. If the scale result survives here, run June 2022 for a
# genuinely independent test:
#     START=2022-06-01 END=2022-06-30 TAG=scale2022jun \
#         bash slurm/submit_scale_ladder_31day.sh
#
# Cost: 3 arms x 5 folds = 15 GPU tasks, two concurrent per array, roughly
# 37 min/fold at 31 days. The array walltime is 48 h, so there is ample margin.
#
# Usage
#   bash slurm/submit_scale_ladder_31day.sh
#   DRY_RUN=1 bash slurm/submit_scale_ladder_31day.sh

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"

export START="${START:-2022-05-01}"
export END="${END:-2022-05-31}"
export TAG="${TAG:-scale2022}"
export MEMBERS="${MEMBERS:-30}"
export ARMS="${ARMS:-RAW S04 S08}"

echo "============================================================"
echo " Scale-ladder confirmation"
echo "   window   $START .. $END"
echo "   arms     $ARMS"
echo "   members  $MEMBERS, five rotated folds"
echo "   target   ~1240 withheld station-days (3.3x the 10-day screen)"
echo "   outputs  data/processed/bmd_imerg_eval_${TAG}_*/"
echo "============================================================"
echo

# Everything below -- cutting the window out of the monthly archive, coarsening
# to 0.4 and 0.8 degrees, and moving factor/error_corr_cells/stride together --
# is already in the main submitter. Duplicating it here would be a second place
# for those four coupled settings to drift apart, which is exactly the mistake
# that killed the first scale arms.
bash slurm/submit_ingestion_experiment.sh

cat <<EOF

 When these finish:

   TAG=$TAG bash slurm/run_ingestion_report.sh

 Read, in order:
   1. n_wet -- expect ~580 against the 178 of the ten-day screen
   2. delta for S04 and S08. THIS is the test: at 3.3x the sample the S04
      interval should be ~0.18 wide if the effect is real and stable.
   3. pattern correlation and wet area across RAW -> S04 -> S08. The ten-day
      run gave 0.50/0.59/0.67 and 0.922/0.875/0.820. If the ordering holds at
      a month, the scale result is solid.

 Decide as follows, fixed in advance:
   * S04 or S08 delta excludes zero        -> coarse assimilation is the
                                              production configuration
   * intervals still span zero BUT the structural ordering holds
                                           -> report as a structure result;
                                              gauges-only stays production
   * ordering does not hold                -> the ten-day trend was noise
EOF
