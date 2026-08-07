#!/bin/bash
#SBATCH --job-name=bdhires-cpc-satellite
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# CPC (0.5 deg) as the pseudo-satellite, real BMD gauges, 2024-05-01..05.
# Fixed: error_corr_cells units (1.0 = footprints, not fine cells)
# Fixed: day-shift alignment declared in CPC file attributes
# Fixed: _smooth2d clamping prevents kernel shrinking on coarse grids
#
# READ scripts/34_make_cpc_pseudo_satellite.py BEFORE INTERPRETING THIS.
# CPC conditions the prior and is itself a gauge analysis, so the withheld-gauge
# scores are NOT "how a 0.5 degree satellite would perform". What this validly
# tests is footprint SCALE: stride-1 IMERG (0.1 deg) diverged while stride-3
# (0.3 deg effective) was stable, and CPC at 0.5 deg is coarser than either.
#
# Two arms, matching the IMERG confidence test so the numbers line up:
#   production  stride 3, R multiplier 1.0   (the IMERG settings that worked)
#   measured    stride 1, R multiplier 1.0   (every footprint -- the setting that
#                                             broke IMERG)
#
# At 0.5 deg there are only 12x12 footprints over the domain, so stride 3 leaves
# ~16 observations a day. If 'measured' is stable here where it diverged for
# IMERG, the instability is about observation DENSITY, not about IMERG.

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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bdhires-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" logs

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: This job must run on an aarch64 Grace node."
    exit 1
}

START="${CPC_START:-2024-05-01}"
END="${CPC_END:-2024-05-05}"
MEMBERS="${CPC_MEMBERS:-30}"
FOLDS="${CPC_FOLDS:-5}"
FOLD="${CPC_FOLD:-0}"
# 8, NOT 10. 128 / 8 = 16 exactly, so the footprint grid covers the whole domain
# (top edge 26.7N) and every BMD station is inside it. Factor 10 crops 128 -> 120
# cells, dropping the domain top to 26.3N, which puts Tetulia (26.583N) genuinely
# outside -- no coordinate tolerance can or should rescue that. The cost is 0.4 deg
# footprints instead of 0.5, which is still far coarser than IMERG's 0.1/0.3 and
# so still answers the footprint-scale question.
FACTOR="${CPC_FACTOR:-8}"
DAY_SHIFT="${CPC_DAY_SHIFT:--1}"
CKPT="${BMD_CKPT:-runs/prior_h100_cpc/best.pt}"
CONFIG="${BMD_CONFIG:-configs/da_cpc_satellite.yaml}"
OFFSET="${BACKGROUND_DAY_OFFSET:--1}"
ZARR="${CPC_ZARR:-data/processed/bd_wide_cpc.zarr}"

ROOT="${CPC_ROOT:-data/processed/cpc_satellite}"
mkdir -p "$ROOT"
TAG="${START//-/}_${END//-/}"

STATIONS="$ROOT/bmd_${TAG}.csv"
PSEUDO="$ROOT/cpc_pseudo_${TAG}.nc"

BMD_DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
BMD_STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"

echo "CPC pseudo-satellite DA: $START to $END, $MEMBERS members, factor $FACTOR"
echo "Config $CONFIG (factor must match $FACTOR), day shift $DAY_SHIFT"

# ------------------------------------------------------------------ gauges
if [[ ! -s "$STATIONS" ]]; then
    "$PYTHON_BIN" -u scripts/05_convert_bmd_dir.py \
        --data-dir "$BMD_DATA_DIR" \
        --stations "$BMD_STATIONS" \
        --start "$START" --end "$END" \
        --out "$STATIONS" \
        --summary "${STATIONS%.csv}_stations.csv" \
        --report "${STATIONS%.csv}_qc.json"
else
    echo "Reusing $STATIONS"
fi

# -------------------------------------------------- CPC as pseudo-satellite
# The day shift is the part most likely to be got wrong: CPC is a 00-00 UTC
# calendar day, BMD day D is the 24 h ending 03:00 UTC on D. Measured over
# 43,781 station-days CPC peaks against the gauges at lag -1 (0.756) versus
# 0.387 at lag 0.
# Reuse only if the file is actually USABLE, not merely present. A stale file
# written before the schema changed will otherwise be reused forever and fail
# downstream with a confusing error from the loader instead of here.
REGENERATE=1
if [[ -s "$PSEUDO" ]]; then
    if "$PYTHON_BIN" - "$PSEUDO" "$FACTOR" "$DAY_SHIFT" <<'PY'
import sys
import xarray as xr

path, factor, shift = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with xr.open_dataset(path) as dataset:
    attrs = dataset.attrs
    checks = {
        "bmd_window_alignment == 'day-shift'":
            str(attrs.get("bmd_window_alignment", "")) == "day-shift",
        "bmd_accumulation_end_hour_utc == 3":
            int(attrs.get("bmd_accumulation_end_hour_utc", -1)) == 3,
        f"block_factor == {factor}":
            int(attrs.get("block_factor", -1)) == factor,
        f"day_shift_applied == {shift}":
            int(attrs.get("day_shift_applied", 999)) == shift,
    }
bad = [name for name, ok in checks.items() if not ok]
if bad:
    print("        stale/mismatched: " + "; ".join(bad))
    raise SystemExit(1)
PY
    then
        REGENERATE=0
    else
        echo "    $PSEUDO does not match the requested settings; regenerating."
    fi
fi

if [[ "$REGENERATE" == "1" ]]; then
    "$PYTHON_BIN" -u scripts/34_make_cpc_pseudo_satellite.py \
        --zarr "$ZARR" \
        --start "$START" --end "$END" \
        --factor "$FACTOR" \
        --day-shift "$DAY_SHIFT" \
        --out "$PSEUDO" \
        --report "${PSEUDO%.nc}_qc.json"
else
    echo "Reusing $PSEUDO (attributes verified)"
fi

# -------------------------------------------------------------------- arms
# Arm spec: NAME:STRIDE:RMULT[:key=value,key=value]
# The optional fourth field is passed to scripts/15 as --set, so an experiment
# never depends on hand-editing the YAML between submissions. That failure mode
# already cost three arms: measured_R, T1p0 and s1r1 came back byte-identical
# because the edits behind them were never made, and each job still ran for an
# hour and wrote a plausible dump.
ARMS="${CPC_ARMS:-production:3:1.0 measured:1:1.0}"

# Reuse is a trap when a config changes: the old guard skipped any arm whose NPZ
# existed, so a re-submitted arm replayed its cached result. Set CPC_FORCE=1 to
# regenerate, and note the provenance block recorded in every dump now makes a
# stale reuse visible in scripts/36 rather than silent.
FORCE="${CPC_FORCE:-0}"

for spec in $ARMS; do
    IFS=':' read -r NAME STRIDE RMULT OVERRIDES <<< "$spec"
    PREFIX="$ROOT/${NAME}_${TAG}"
    echo
    echo "=== arm '$NAME': stride $STRIDE, R multiplier $RMULT ==="
    if [[ -n "${OVERRIDES:-}" ]]; then
        echo "    config overrides: $OVERRIDES"
    fi

    SET_ARGS=()
    if [[ -n "${OVERRIDES:-}" ]]; then
        IFS=',' read -ra PAIRS <<< "$OVERRIDES"
        for pair in "${PAIRS[@]}"; do
            SET_ARGS+=(--set "$pair")
        done
    fi

    if [[ -s "${PREFIX}.npz" && "$FORCE" != "1" ]]; then
        echo "Reusing ${PREFIX}.npz  (CPC_FORCE=1 to regenerate)"
        if [[ -n "${OVERRIDES:-}" ]]; then
            echo "    WARNING: this arm requests config overrides but is being"
            echo "    REUSED. If the cached dump was written before those overrides"
            echo "    existed it does NOT reflect them. Re-run with CPC_FORCE=1."
        fi
    else
        if [[ "$FORCE" == "1" ]]; then rm -f "${PREFIX}.npz"; fi
        "$PYTHON_BIN" -u scripts/15_bmd_month_example.py \
            --config "$CONFIG" \
            --ckpt "$CKPT" \
            --stations "$STATIONS" \
            --imerg "$PSEUDO" \
            --start "$START" --end "$END" \
            --members "$MEMBERS" \
            --holdout-folds "$FOLDS" --holdout-fold "$FOLD" \
            --background-day-offset "$OFFSET" \
            --imerg-stride "$STRIDE" \
            --imerg-r-multiplier "$RMULT" \
            "${SET_ARGS[@]}" \
            --out "${PREFIX}.npz" \
            --report "${PREFIX}.json"
    fi
done

# Gauge-truth evaluation over every arm at once. scripts/33 is deliberately NOT
# run here: it scores against CHIRPS, which is the weakest of the three products
# against BMD gauges at daily scale and runs 44 mm/day low above 50 mm/day.
"$PYTHON_BIN" -u scripts/36_gauge_truth_da_evaluation.py \
    --dumps "$ROOT"/*.npz \
    --stats "${BMD_STATS:-data/processed/stats_cpc.json}" \
    --sigma-rep "${CPC_SIGMA_REP:-0.410}" \
    --out-dir "$ROOT/gauge_truth_${TAG}"

echo
echo "Done. Compare against the IMERG run at the same settings:"
echo "  CPC   : $ROOT/{production,measured}_${TAG}_impact.json"
echo "  IMERG : data/processed/real_obs_confidence/{production,measured}_${TAG}_impact.json"
echo
echo "Read MEDIAN columns. And remember CPC is not independent of the prior --"
echo "small increments here may mean 'already knew that', not 'observation agrees'."
