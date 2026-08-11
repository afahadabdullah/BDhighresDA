#!/bin/bash
#SBATCH --job-name=bdhires-osse-bmd
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%2
#SBATCH --output=logs/%x-%A_%a.out
#
# CHIRPS OSSE on the real BMD geometry. PERFECT OBSERVATIONS FIRST.
#
# CHIRPS over 2021-2024 is the 0.05-degree nature run. It supplies BOTH
# observation types, so they are mutually consistent by construction:
#   pseudo-gauges    CHIRPS sampled at the real BMD station coordinates
#   pseudo-IMERG     CHIRPS averaged onto exact nested 0.1-degree footprints
# Everything is scored on WITHHELD pseudo-gauges: 8 of ~40, matching the
# real-data folds so the two experiments can be read against each other.
#
# WHY PERFECT FIRST
# The previous OSSE ran only the noisy arms, and the pseudo-satellite scored
# -68% CRPS against its own background. That was not a property of satellite
# information: the correlation inflation applied to R was ALSO being used to
# manufacture the observation error, so the synthetic satellite carried a
# spatially correlated error of sd 0.91 in transformed units against the
# gauges' white 0.269 (see docs/ablation_pseudo_satellite.tex).
#
# With --obs-error exact neither stream is perturbed at all. The code passes
# CHIRPS values through unchanged and asserts the invariant that the
# pseudo-satellite equals the same-day CHIRPS block mean. Arms 0-2 therefore
# measure what the assimilation can do with correct information, which is the
# question that has to be answered before any error sensitivity means anything.
#
# WHY NO CORRELATION CONTROL ON THE EXACT ARMS
# --satellite-correlation-control inflates R by 2*pi*(corr/stride)^2 to avoid
# treating correlated observation ERRORS as independent evidence. With exact
# observations there is no error to be correlated, so the inflation would only
# down-weight perfect information. It is therefore off for arms 0-2 and on for
# the noisy sensitivity arms, where it is doing its actual job.
#
# ARMS
#   0  gauges_exact_bmd          gauges only, no error
#   1  satellite_exact_bmd       0.1-degree footprints only, no error
#   2  simultaneous_exact_bmd    both, no error            <- primary
#   --- sensitivities, not the headline; submit with ARRAY=3-6 -------------
#   3  gauges_realistic_bmd      gauges only, configured error
#   4  satellite_realistic_bmd   footprints only, configured + correlated error
#   5  simultaneous_realistic_bmd both, configured error
#   6  gauges_tune_bmd           gamma x sigma_obs x prior_temperature, 60 combos
#   --- better gauge ingestion, exact observations (ARRAY=7-9) --------------
#   7  gauges_exact_nbr          exact gauges, NEIGHBOURED holdout
#   8  gauges_tune_exact_nbr     the same, sweeping the ingestion settings
#   9  gauges_exact_spread_ref   exact gauges, ISOLATED holdout, kept as the
#                                honest comparator for arm 7
#
# WHY ARMS 7-9
# spread_holdout seeds at the station farthest from the centroid and then
# samples farthest-point, so it selects the most ISOLATED stations by
# construction. On a BMD-like geometry it put all three isolates into an
# eight-station holdout, with a mean nearest-neighbour distance of 90 km
# against 32 km for a neighbour-supported selection. A gauge with nothing
# inside the ~146 km variogram range cannot be reconstructed from
# observations, so its score measures the PRIOR, not the assimilation.
#
# Two defects are excluded, and they fail differently. ISOLATION: no neighbour
# within 75 km, so nothing to reconstruct from. EDGE: neighbours on one side
# only, so reconstruction is extrapolation rather than interpolation, caught by
# the largest angular gap between bearings to the nearest stations. On BMD-like
# geometry the old holdout reached 291 km to the nearest ASSIMILATED station and
# a 322 degree gap; the new one, 55 km and 186 degrees.
#
# Support is checked against the ASSIMILATED stations, not all stations: a
# holdout can otherwise strip the very neighbours that qualified its members,
# each supporting the next and all removed together.
#
# Arm 7 withholds only stations that pass both tests. This RAISES
# apparent skill, so arm 9 repeats the old isolated holdout unchanged and both
# are reported: the pair is the result, either alone is cherry-picking.
#
# WHY 2021-2024 AND NOT 2020: the prior splits train [1981,2018],
# val [2019,2020], test [2021,2025]. Scoring on 2020 would use years the
# checkpoint selection saw.

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: not found: $PYTHON_BIN"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"

# These are aarch64 Grace nodes. Guard explicitly: the failure mode otherwise
# is an x86_64 binary on PATH dying with "cannot execute binary file", which
# reads as a corrupt install rather than a wrong architecture.
[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: this job must run on an aarch64 Grace node (got $(uname -m))."
    exit 1
}

# srun inside an allocation fails on the GH200 nodes with "CPU binding outside
# of job step allocation ... Unable to satisfy cpu bind request": since SLURM
# 22.05 the step does not inherit --cpus-per-task, so its default binding asks
# for more than the 8 CPUs the job holds. slurm/bmd_imerg_example.sbatch
# already carries this fix; running Python directly is the working path here
# too. Set BDHIRES_LAUNCH="srun --cpu-bind=none" to restore a step launcher.
read -r -a LAUNCH <<< "${BDHIRES_LAUNCH:-}"
run_step() { "${LAUNCH[@]+"${LAUNCH[@]}"}" "$@"; }

CKPT="${OSSE_CKPT:-runs/prior_h100_cpc/best.pt}"
CONFIG="${OSSE_CONFIG:-configs/da.yaml}"
DAYS="${OSSE_BMD_DAYS:-48}"
MEMBERS="${OSSE_BMD_MEMBERS:-16}"
# Arms 7-9 use 50 members. scripts/41 measured fair CRPS to be flat in ensemble
# size while the member-sampling interval narrows from ~3.0 mm/day at m=2 to
# ~0.3 at m=30 and ~0 at m=50: members buy PRECISION, not skill. These three
# arms are compared against each other and, in arm 8, ranked sixty ways, so
# precision is exactly what they need.
MEMBERS_INGESTION="${OSSE_BMD_MEMBERS_INGESTION:-50}"
START="${OSSE_BMD_START:-2021-01-01}"
END="${OSSE_BMD_END:-2024-12-31}"
WITHHOLD="${OSSE_BMD_WITHHOLD:-0.2}"
SEED="${OSSE_BMD_SEED:-7}"
ROOT="${OSSE_BMD_ROOT:-data/processed/osse_bmd}"
NETWORK="${OSSE_BMD_NETWORK:-bmd}"
FACTOR="${OSSE_BMD_FACTOR:-2}"          # 2 x 0.05 deg = the 0.1 deg IMERG footprint
# 8 rather than 16 for the 50-member tuning arm. At 16 days it is 48,000
# member-days against 768 for a 16-member single-config arm -- 62x -- which
# does not fit the wall. Halving the days keeps the RANKING, which is what a
# sweep is for, while a single configuration still gets all 48 days.
TUNE_DAYS="${OSSE_BMD_TUNE_DAYS:-8}"

# The 'bmd' token reads coordinates from this catalogue. scripts/10_osse.py
# defaults to data/bmd/Stations.csv, which does not exist in this checkout.
CATALOG="${OSSE_BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
[[ -s "$CATALOG" ]] || {
    echo "ERROR: BMD station catalogue not found: $CATALOG"; exit 1; }

#        LABEL                        MODE         ERROR      STRIDE CORRCTL TUNE
case "$SLURM_ARRAY_TASK_ID" in
    0) SET=("gauges_exact_bmd"           "gauges"      "exact"     1 0 0) ;;
    1) SET=("satellite_exact_bmd"        "satellite"   "exact"     1 0 0) ;;
    2) SET=("simultaneous_exact_bmd"     "combined"    "exact"     1 0 0) ;;
    3) SET=("gauges_realistic_bmd"       "gauges"      "realistic" 3 0 0) ;;
    4) SET=("satellite_realistic_bmd"    "satellite"   "realistic" 3 1 0) ;;
    5) SET=("simultaneous_realistic_bmd" "combined"    "realistic" 3 1 0) ;;
    6) SET=("gauges_tune_bmd"            "gauges"      "realistic" 3 0 1) ;;
    7) SET=("gauges_exact_nbr"           "gauges"      "exact"     1 0 0) ;;
    8) SET=("gauges_tune_exact_nbr"      "gauges"      "exact"     1 0 1) ;;
    9) SET=("gauges_exact_spread_ref"    "gauges"      "exact"     1 0 0) ;;
    *) echo "ERROR: unknown task $SLURM_ARRAY_TASK_ID"; exit 2 ;;
esac
LABEL="${SET[0]}"; MODE="${SET[1]}"; ERROR="${SET[2]}"
STRIDE="${SET[3]}"; CORRCTL="${SET[4]}"; TUNE="${SET[5]}"

# Arms 7 and 8 verify only where verification is meaningful; arm 9 is the
# unrestricted comparator that keeps arm 7 honest.
HOLDOUT="${OSSE_BMD_HOLDOUT:-spread}"
case "$SLURM_ARRAY_TASK_ID" in
    7|8) HOLDOUT="neighbored" ;;
esac
NEIGHBOR_KM="${OSSE_BMD_NEIGHBOR_KM:-75}"
MAX_GAP_DEG="${OSSE_BMD_MAX_GAP_DEG:-200}"

case "$SLURM_ARRAY_TASK_ID" in
    7|8|9) MEMBERS="$MEMBERS_INGESTION" ;;
esac

OUT="$ROOT/$LABEL"
mkdir -p "$OUT"

echo "OSSE arm: $LABEL"
echo "  mode=$MODE error=$ERROR stride=$STRIDE corr_control=$CORRCTL tune=$TUNE"
FOOTPRINT_DEG="$(awk -v f="$FACTOR" 'BEGIN{printf "%.2f", f*0.05}')"
echo "  network=$NETWORK ($CATALOG) footprint factor=$FACTOR (= $FOOTPRINT_DEG deg)"
echo "  window=$START..$END days=$DAYS members=$MEMBERS withhold=$WITHHOLD seed=$SEED"
echo "  holdout=$HOLDOUT (neighbour <= ${NEIGHBOR_KM} km, bearing gap <= ${MAX_GAP_DEG} deg)"
if [[ "$TUNE" == "1" ]]; then
    COMBOS=60   # gamma(3) x sigma_obs(4) x prior_temperature(5), configs/da.yaml
    WORK=$(( TUNE_DAYS * COMBOS * MEMBERS ))
    echo "  TUNING: $TUNE_DAYS days x $COMBOS combinations x $MEMBERS members"
    echo "          = $WORK member-days, against $(( DAYS * MEMBERS )) for a"
    echo "          single-configuration arm. Reduce OSSE_BMD_TUNE_DAYS if this"
    echo "          approaches the 24 h wall."
else
    echo "  work: $(( DAYS * MEMBERS )) member-days"
fi
if [[ "$ERROR" == "exact" ]]; then
    echo "  PERFECT OBSERVATIONS: no error is added to either stream, and the"
    echo "  run asserts the pseudo-satellite equals the CHIRPS block mean."
fi

EXTRA=()
[[ "$MODE" != "gauges" ]] && EXTRA+=(--satellite-stride "$STRIDE"
                                     --satellite-factor "$FACTOR")
[[ "$CORRCTL" == "1" ]] && EXTRA+=(--satellite-correlation-control)
# --tune-network is REQUIRED: without it --tune selects the largest SYNTHETIC
# network and silently ignores the BMD geometry the experiment is about.
[[ "$TUNE" == "1" ]] && EXTRA+=(--tune --tune-network "$NETWORK"
                                --tune-days "$TUNE_DAYS")

run_step "$PYTHON_BIN" -u scripts/10_osse.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --start "$START" \
    --end "$END" \
    --month 0 \
    --days "$DAYS" \
    --members "$MEMBERS" \
    --networks "$NETWORK" \
    --bmd-stations "$CATALOG" \
    --station-layout spread \
    --holdout-layout "$HOLDOUT" \
    --holdout-neighbor-km "$NEIGHBOR_KM" \
    --holdout-max-gap-deg "$MAX_GAP_DEG" \
    --withhold "$WITHHOLD" \
    --seed "$SEED" \
    --obs-error "$ERROR" \
    --observation-mode "$MODE" \
    --dump "$OUT/ensemble.npz" \
    --dump-network "$NETWORK" \
    --dump-obs-error "$ERROR" \
    --out-figure "$OUT/osse_overview.png" \
    --out-report "$OUT/osse_report.json" \
    "${EXTRA[@]+"${EXTRA[@]}"}"

if [[ "$TUNE" == "1" ]]; then
    echo "Tuning arm complete: $OUT/osse_report.json holds the 60-combination"
    echo "matrix. Downscaling claims are not rebuilt for it -- they describe one"
    echo "configuration and this arm deliberately has sixty."
    exit 0
fi

run_step "$PYTHON_BIN" -u scripts/11_da_diagnostics.py \
    --dump "$OUT/ensemble.npz" \
    --out-figure "$OUT/da_diagnostics.png" \
    --out-report "$OUT/da_diagnostics.json"

run_step "$PYTHON_BIN" -u scripts/14_osse_summary.py \
    --dump "$OUT/ensemble.npz" \
    --cases 4 \
    --out-reconstruction "$OUT/reconstruction.png" \
    --out-matrix "$OUT/scale_matrix.png" \
    --out-report "$OUT/scale_summary.json"

# Claim A: does the prior beat the 0.5-degree coarse input? Scored on the
# BACKGROUND only, so no assimilated observation counts as downscaling skill.
# Claim B: does the analysis beat a null handed the truth's own 0.1-degree
# block means? In the satellite_exact arm the assimilated observation IS that
# null's information, so Claim B there measures precisely what the prior adds
# BELOW the footprint -- the cleanest form of the question in this project.
run_step "$PYTHON_BIN" -u scripts/22_evaluate_osse_downscaling.py \
    --dump "$OUT/ensemble.npz" \
    --label "$LABEL" \
    --minimum-valid-fraction 1.0 \
    --out-figure "$OUT/downscaling_diagnostics.png" \
    --out-report "$OUT/downscaling.json" \
    --out-curve-data "$OUT/downscaling_curves.npz" \
    --out-spatial-data "$OUT/spatial_fields.nc"

echo "Arm $LABEL complete: $OUT"
