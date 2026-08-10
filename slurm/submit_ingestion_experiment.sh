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
# scripts/15 compares the IMERG time axis to the checkpoint dates with
# array_equal, so the prepared file must span the window and nothing else. The
# monthly archive therefore cannot be handed over directly even though it holds
# exactly the right days.
#
# It does NOT need rebuilding from half-hourly granules. The monthly files were
# written by download_imerg_halfhourly_2021_2024.sbatch with the same
# --source-frequency half-hourly --min-count 48 --accumulation-end-hour-utc 3
# that a rebuild would use, so the accumulation is already correct and only
# needs cutting. scripts/43 does the cut and re-checks every attribute
# scripts/15 validates, which takes seconds instead of an hour and does not
# depend on granules that may since have been cleaned up.
#
# The cut happens ONCE, here, into a shared location that every arm reuses.
# Doing it per-arm would reintroduce the concurrent-write race that killed two
# folds of the last screen.
SHARED_DIR="data/processed/imerg_prepared_${TAG}"
ARCHIVE_GLOB="${IMERG_ARCHIVE_GLOB:-data/processed/imerg_bd_aligned_*.nc}"
START_CLEAN="${START//-/}"; END_CLEAN="${END//-/}"
export IMERG_PREPARED="$SHARED_DIR/imerg_aligned_${START_CLEAN}_${END_CLEAN}.nc"
export IMERG_QC="$SHARED_DIR/imerg_aligned_${START_CLEAN}_${END_CLEAN}_qc.json"
export IMERG_REUSE_PREPARED=1
mkdir -p "$SHARED_DIR"

if [[ ! -s "$IMERG_PREPARED" ]]; then
    echo "Cutting $START..$END out of the prepared archive ($ARCHIVE_GLOB)"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  DRY RUN: would run scripts/43_subset_prepared_imerg.py"
    elif ! PYTHONPATH="$PWD/src" python -u scripts/43_subset_prepared_imerg.py \
            --input "$ARCHIVE_GLOB" \
            --start "$START" --end "$END" \
            --out "$IMERG_PREPARED" --report "$IMERG_QC"; then
        echo
        echo "============================================================"
        echo " The archive does not cover $START..$END, so the days have to be"
        echo " built from half-hourly granules before anything can be submitted:"
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
fi
[[ "$DRY_RUN" == "1" ]] || echo "Using prepared IMERG: $IMERG_PREPARED"

# ------------------------------------------------------ coarsened scale ladder
# observations.imerg.factor is NOT an operator knob. scripts/15 builds
# grid.lat[:n*factor].reshape(n,factor).mean(1) and requires the observation
# FILE's latitudes to equal it, so factor declares which grid the file is on.
# Setting factor=8 against the 0.1 deg file killed both scale arms with
#   ValueError: operands could not be broadcast together with shapes (64,) (16,)
# The 0.4 and 0.8 degree arms therefore need their own files, and three other
# settings must move with them (see scripts/44 for the arithmetic).
COARSE_04="$SHARED_DIR/imerg_0p4deg_${START_CLEAN}_${END_CLEAN}.nc"
COARSE_08="$SHARED_DIR/imerg_0p8deg_${START_CLEAN}_${END_CLEAN}.nc"

coarsen() {   # $1 target factor, $2 output path
    [[ -s "$2" ]] && { echo "Reusing coarsened IMERG: $2"; return 0; }
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  DRY RUN: would coarsen to factor $1 -> $2"; return 0
    fi
    PYTHONPATH="$PWD/src" python -u scripts/44_coarsen_imerg_observations.py \
        --input "$IMERG_PREPARED" --factor "$1" --out "$2"
}

WANTED_PRE="${ARMS:-}"
if [[ -z "$WANTED_PRE" || " $WANTED_PRE " == *" S04 "* ]]; then
    coarsen 8 "$COARSE_04"
fi
if [[ -z "$WANTED_PRE" || " $WANTED_PRE " == *" S08 "* ]]; then
    coarsen 16 "$COARSE_08"
fi

# ------------------------------------------------------------------------ arms
# Format: TAG | IMERG_STRIDE | IMERG_R_MULTIPLIER | BMD_SET | IMERG_FILE
#         (BMD_SET semicolon-separated; IMERG_FILE empty = the 0.1 deg default)
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
#
# The scale arms carry FOUR coupled changes, not one. Changing only `factor`
# is what killed them the first time; changing only the file would leave the
# operator on the wrong grid; leaving error_corr_cells at 3.0 would assert a
# 1.2 deg correlation length instead of the configured 0.30 deg and inflate R
# by its square; and leaving stride at 3 would reduce 0.8 deg to SEVEN usable
# footprints. error_corr_cells is denominated in cells OF THE OBSERVATION GRID,
# so holding the physical length fixed means 3.0 -> 0.75 -> 0.375.
declare -a ARM_SPECS=(
  # --- scale ladder: the same product at three footprint sizes --------------
  "RAW|3|1.0||"
  "S04|1|1.0|observations.imerg.factor=8;observations.imerg.error_corr_cells=0.75|$COARSE_04"
  "S08|1|1.0|observations.imerg.factor=16;observations.imerg.error_corr_cells=0.375|$COARSE_08"

  # --- gauge strength: never varied before, and the Desroziers ratio of 0.08
  #     says the analysis fits gauges ~12x harder than R permits -------------
  "GW|3|1.0|observations.gauges.sigma_obs=0.05|"
  "GL|3|1.0|observations.gauges.sigma_obs=0.25|"
  "GM|3|1.0|observations.gauges.sigma_obs=0.41|"

  # --- satellite strength, via sigma_obs rather than the R multiplier -------
  "SW|3|1.0|observations.imerg.sigma_obs=0.20|"
  "SL|3|1.0|observations.imerg.sigma_obs=1.00|"

  # --- the ratio between the two streams, which is the actual trade-off ----
  "RATIO|3|1.0|observations.gauges.sigma_obs=0.05;observations.imerg.sigma_obs=1.00|"

  # --- measured observation error from scripts/35 instead of assumed -------
  "MEASR|3|1.0|observations.gauges.representativeness=0.410;observations.imerg.representativeness=0.419|"

  # --- one thinning contrast, to confirm the earlier null on a new window --
  "S1|1|1.0||"
)

WANTED="${ARMS:-}"
SUBMITTED=0

for spec in "${ARM_SPECS[@]}"; do
    IFS='|' read -r NAME STRIDE RMULT SET ARM_IMERG <<< "$spec"
    if [[ -n "$WANTED" ]] && [[ " $WANTED " != *" $NAME "* ]]; then
        continue
    fi
    ARM_IMERG="${ARM_IMERG:-$IMERG_PREPARED}"
    LABEL="${TAG}_${NAME}"
    echo
    echo "=== arm $NAME  (stride $STRIDE, R x$RMULT)"
    [[ -n "$SET" ]] && echo "    overrides: $SET"
    [[ "$ARM_IMERG" != "$IMERG_PREPARED" ]] && echo "    observations: $ARM_IMERG"
    echo "    label: $LABEL -> data/processed/bmd_imerg_eval_${LABEL}/"
    if [[ "$DRY_RUN" == "1" ]]; then
        continue
    fi
    # An arm pointed at a file that was never built would fall through to the
    # regeneration path in the sbatch and rebuild it from half-hourly granules
    # under a different name -- silently, in five racing array tasks.
    if [[ ! -s "$ARM_IMERG" ]]; then
        echo "    ERROR: $ARM_IMERG does not exist; skipping $NAME" >&2
        continue
    fi
    IMERG_STRIDE="$STRIDE" \
    IMERG_R_MULTIPLIER="$RMULT" \
    BMD_SET="$SET" \
    IMERG_PREPARED="$ARM_IMERG" \
    IMERG_QC="${ARM_IMERG%.nc}_qc.json" \
    IMERG_REUSE_PREPARED=1 \
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

 When the runs finish, start with the question the experiment is about --
 does adding the satellite to the gauges help? Both arms come from the same
 dump, so the pairing is exact:

   python scripts/42_select_best_config.py \\
       --dumps 'data/processed/bmd_imerg_eval_${TAG}_*/*.npz' \\
       --arm combined --vs-arm gauges --reference ${TAG}_RAW \\
       --out-dir data/processed/${TAG}_selection_combined

 Then the other two arms, to see what the satellite contributes alone and
 what the background already had:

   for a in gauges satellite; do
     python scripts/42_select_best_config.py \\
         --dumps 'data/processed/bmd_imerg_eval_${TAG}_*/*.npz' \\
         --arm \$a --vs-arm background --reference ${TAG}_RAW \\
         --out-dir data/processed/${TAG}_selection_\$a
   done

 Read in this order:
   1. n_wet  -- below ~50 the window is too dry to conclude anything
   2. the combined-minus-gauges interval, NOT the point estimate
   3. the between-configuration intervals against ${TAG}_RAW
   4. wet-area inside/outside the product envelope
   5. pattern correlation against CHIRPS/IMERG/CPC

 Then, for structure and monthly means:
   python scripts/38_multiyear_gauge_evaluation.py \\
       --dumps 'data/processed/bmd_imerg_eval_${TAG}_RAW/*.npz' \\
       --out-dir data/processed/${TAG}_figures
   python scripts/40_spatial_structure_screen.py \\
       --dumps 'data/processed/bmd_imerg_eval_${TAG}_*/*fold0*.npz' \\
       --arm combined --out-dir data/processed/${TAG}_structure
============================================================
EOF
