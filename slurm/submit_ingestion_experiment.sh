#!/usr/bin/env bash
# Ingestion experiment: how should IMERG and BMD be combined?
#
# Design and reasoning: docs/EXPERIMENT_ingestion.md
#
# Primary target   withheld BMD stations (the only direct measurement)
# Secondary target spatial structure against the CPC/IMERG/CHIRPS envelope
# Fixed            the prior (runs/prior_h100_cpc/best.pt); nothing is retrained
# Window           2022-05-01..10, five rotated folds, 30 members  (~370
#                  withheld station-days, matching the May 2024 screen)
#
# Every arm below is CONFIG-ONLY and runs with the code as it stands. The three
# arms that need new code -- QM (bias-corrected IMERG), GAP (satellite only away
# from gauges) and MULTI (two footprint scales at once) -- are listed at the
# bottom and are deliberately NOT submitted here, so this script never half-runs
# an experiment.
#
# There is no separate "gauges only" arm on purpose. Every run already writes
# four arms -- background, gauges, satellite, combined -- so gauges-only is read
# out of any run with `scripts/42 --arm gauges`. Submitting it again would burn
# a fifth of the budget reproducing a field we already have. For the gauge
# STRENGTH arms this matters twice over: their `gauges` arm is exactly the
# quantity under test.
#
# Usage
#   bash slurm/submit_ingestion_experiment.sh              # submit everything
#   ARMS="S04 S08" bash slurm/submit_ingestion_experiment.sh   # a subset
#   DRY_RUN=1 bash slurm/submit_ingestion_experiment.sh    # print, submit nothing

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

START="${START:-2022-05-01}"
END="${END:-2022-05-10}"
MEMBERS="${MEMBERS:-30}"
TAG="${TAG:-ing2022}"
DRY_RUN="${DRY_RUN:-0}"

export BMD_CKPT="${BMD_CKPT:-runs/prior_h100_cpc/best.pt}"
export BMD_CONFIG="${BMD_CONFIG:-configs/da.yaml}"
export BMD_MEMBERS="$MEMBERS"

# ---------------------------------------------------------------- prepared IMERG
# scripts/15 requires the prepared file's dates to match the requested window
# EXACTLY, so the 2021-2024 archive cannot serve a ten-day request. The file is
# therefore built once, up front, into a shared location, and every arm reuses
# it. Building it per-arm would both waste an hour each and reintroduce the
# concurrent-write race that killed two folds of the last screen.
SHARED_DIR="data/processed/imerg_prepared_${TAG}"
START_CLEAN="${START//-/}"; END_CLEAN="${END//-/}"
export IMERG_PREPARED="$SHARED_DIR/imerg_aligned_${START_CLEAN}_${END_CLEAN}.nc"
export IMERG_QC="$SHARED_DIR/imerg_aligned_${START_CLEAN}_${END_CLEAN}_qc.json"
export IMERG_REUSE_PREPARED=1
mkdir -p "$SHARED_DIR"

if [[ ! -s "$IMERG_PREPARED" ]]; then
    echo "============================================================"
    echo " Prepared IMERG for $START..$END is missing."
    echo " Build it ONCE before submitting, so the arms cannot race:"
    echo
    echo "   python scripts/08_prepare_imerg_observations.py \\"
    echo "       --input data/imerg_halfhourly \\"
    echo "       --source-frequency half-hourly \\"
    echo "       --start $START --end $END \\"
    echo "       --min-count 48 \\"
    echo "       --accumulation-end-hour-utc 3 \\"
    echo "       --out $IMERG_PREPARED \\"
    echo "       --report $IMERG_QC"
    echo
    echo " Then re-run this script."
    echo "============================================================"
    exit 1
fi
echo "Using prepared IMERG: $IMERG_PREPARED"

# ------------------------------------------------------------------------ arms
# Format: TAG | IMERG_STRIDE | IMERG_R_MULTIPLIER | BMD_SET (semicolon-separated)
#
# BMD_SET uses SEMICOLONS, not commas: sbatch --export takes a comma-separated
# list and a comma inside a value silently truncates it.
#
# The R multiplier is 1.0 everywhere on purpose. It was measured to inject noise
# rather than down-weight -- perturb_observations draws with sd = sqrt(R), so
# inflating R also inflates the perturbations, which is why the s1r10T arm of
# the last screen was significantly WORSE rather than merely satellite-free.
# Down-weighting is done through sigma_obs instead (arms SL and RATIO), which
# reaches the same effective weight by a mechanism that does not add noise.
declare -a ARM_SPECS=(
  # --- scale ladder: the same product at three footprint sizes --------------
  "RAW|3|1.0|"
  "S04|3|1.0|observations.imerg.factor=8"
  "S08|3|1.0|observations.imerg.factor=16"

  # --- gauge strength: never varied before, and the Desroziers ratio of 0.08
  #     says the analysis fits gauges ~12x harder than R permits -------------
  "GW|3|1.0|observations.gauges.sigma_obs=0.05"
  "GL|3|1.0|observations.gauges.sigma_obs=0.25"
  "GM|3|1.0|observations.gauges.sigma_obs=0.41"

  # --- satellite strength, via sigma_obs rather than the R multiplier -------
  "SW|3|1.0|observations.imerg.sigma_obs=0.20"
  "SL|3|1.0|observations.imerg.sigma_obs=1.00"

  # --- the ratio between the two streams, which is the actual trade-off ----
  "RATIO|3|1.0|observations.gauges.sigma_obs=0.05;observations.imerg.sigma_obs=1.00"

  # --- measured observation error from scripts/35 instead of assumed -------
  "MEASR|3|1.0|observations.gauges.representativeness=0.410;observations.imerg.representativeness=0.419"

  # --- one thinning contrast, to confirm the earlier null on a new window --
  "S1|1|1.0|"
)

WANTED="${ARMS:-}"
SUBMITTED=0

for spec in "${ARM_SPECS[@]}"; do
    IFS='|' read -r NAME STRIDE RMULT SET <<< "$spec"
    if [[ -n "$WANTED" ]] && [[ " $WANTED " != *" $NAME "* ]]; then
        continue
    fi
    LABEL="${TAG}_${NAME}"
    echo
    echo "=== arm $NAME  (stride $STRIDE, R x$RMULT)"
    [[ -n "$SET" ]] && echo "    overrides: $SET"
    echo "    label: $LABEL -> data/processed/bmd_imerg_eval_${LABEL}/"
    if [[ "$DRY_RUN" == "1" ]]; then
        continue
    fi
    IMERG_STRIDE="$STRIDE" \
    IMERG_R_MULTIPLIER="$RMULT" \
    BMD_SET="$SET" \
    BMD_START="$START" BMD_END="$END" BMD_EVAL_LABEL="$LABEL" \
      bash slurm/submit_bmd_imerg_eval.sh
    SUBMITTED=$((SUBMITTED + 1))
done

echo
echo "============================================================"
if [[ "$DRY_RUN" == "1" ]]; then
    echo " DRY RUN -- nothing submitted."
else
    echo " Submitted $SUBMITTED arm(s), five folds each."
fi
cat <<EOF

 NOT submitted (these need code that does not exist yet):
   QM     bias-corrected IMERG -- scripts/15 never reads
          observations.imerg.bias_correction, so the knob is inert
   GAP    satellite only away from gauges -- needs distance-weighted R
   MULTI  0.1 and 0.4 degree footprints together -- needs a second operator
   CPCOBS CPC as a third stream -- needs the pseudo-satellite wired in, and
          is NOT an independent observation (see docs/EXPERIMENT_ingestion.md)

 When the runs finish, evaluate all three arms of every configuration:

   for a in combined gauges satellite; do
     python scripts/42_select_best_config.py \\
         --dumps 'data/processed/bmd_imerg_eval_${TAG}_*/*.npz' \\
         --arm \$a --reference ${TAG}_RAW \\
         --out-dir data/processed/${TAG}_selection_\$a
   done

 Read in this order:
   1. n_wet  -- below ~50 the window is too dry to conclude anything
   2. the paired-bootstrap interval, NOT the point estimate
   3. wet-area inside/outside the product envelope
   4. pattern correlation against CHIRPS/IMERG/CPC

 Then, for structure and monthly means:
   python scripts/38_multiyear_gauge_evaluation.py \\
       --dumps 'data/processed/bmd_imerg_eval_${TAG}_RAW/*.npz' \\
       --out-dir data/processed/${TAG}_figures
   python scripts/40_spatial_structure_screen.py \\
       --dumps 'data/processed/bmd_imerg_eval_${TAG}_*/*fold0*.npz' \\
       --arm combined --out-dir data/processed/${TAG}_structure
============================================================
EOF
