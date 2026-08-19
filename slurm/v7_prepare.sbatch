#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v7-prepare
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

# V7 STAGE B target archive.  Stage A does NOT read this file -- it trains on
# the coarsened CPCv2 pack that slurm/v7_prepare_meso.sbatch builds, so that it
# runs the v2 code path verbatim.  (An earlier draft of V7 served both stages
# from this one store; that is no longer the design.)
#
# The coarse support here is 0.1 degrees and the factor is 2, so the "coarse"
# arrays are the meso level and the "fine" arrays are the 0.05-degree product
# level.  The two archives nest exactly rather than being kept in step: both
# grids share the origin and the 0.1-degree lattice, so this archive's 120x120
# coarse grid is the first 120x120 of stage A's 128x128.  The interface between
# the stages is a crop, not a regrid.

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: ARM Python not found: $PYTHON_BIN" >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:?submit this job with sbatch from the repository root}"
export PYTHONPATH="$PWD/src"
mkdir -p logs
[[ "$(uname -m)" == "aarch64" ]] || { echo "ERROR: run on an aarch64 Grace node" >&2; exit 1; }

CHIRPS_GLOB="${V7_CHIRPS_GLOB:-data/raw/chirps/chirps_wide_*.nc}"
CPC_GLOB="${V7_CPC_GLOB:-data/raw/cpc/precip.*.nc}"
ERA5_GLOB="${V7_ERA5_GLOB:-data/raw/era5/era5_daily_*.nc}"
STATIC="${V7_STATIC:-data/static/static_wide_cpc.nc}"
OUT="${V7_PREP_OUT:-data/processed/v7/wide_v7.zarr}"
CHUNK_DAYS="${V7_PREP_CHUNK_DAYS:-32}"
OVERWRITE="${V7_PREP_OVERWRITE:-0}"
START="${V7_START:-1981-01-01}"
END="${V7_END:-2024-12-31}"
FACTOR="${V7_FACTOR:-2}"
COARSE_RES="${V7_COARSE_RES:-0.1}"
SMOOTH_ITERS="${V7_SMOOTH_BASE_ITERATIONS:-2}"

for required in scripts/56_build_chirps_subgrid_targets.py "$STATIC"; do
    [[ -e "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done
for label_and_pattern in "CHIRPS|$CHIRPS_GLOB" "CPC|$CPC_GLOB" "ERA5|$ERA5_GLOB"; do
    label="${label_and_pattern%%|*}"; pattern="${label_and_pattern#*|}"
    compgen -G "$pattern" >/dev/null || {
        echo "ERROR: $label pattern matched no files: $pattern" >&2; exit 1; }
done

if [[ -e "$OUT" && "$OVERWRITE" != "1" ]]; then
    COMPLETE="$("$PYTHON_BIN" - "$OUT" "$START" "$END" "$FACTOR" <<'PY'
import sys
import zarr
from bdhires.data import SUBGRID_SCHEMA
try:
    root = zarr.open_group(sys.argv[1], mode="r")
    encoding = dict(root.attrs.get("subgrid_encoding", {}))
    print("1" if (
        root.attrs.get("complete", False)
        and root.attrs.get("schema") == SUBGRID_SCHEMA
        and root.attrs.get("source_start_date") == sys.argv[2]
        and root.attrs.get("source_end_date") == sys.argv[3]
        and int(encoding.get("factor", 0)) == int(sys.argv[4])
    ) else "0")
except Exception:
    print("0")
PY
)"
    if [[ "$COMPLETE" == "1" ]]; then
        echo "V7 archive already complete; reusing $OUT"
        exit 0
    fi
    echo "ERROR: $OUT exists but is not a completed V7 archive" >&2
    echo "Inspect it, or resubmit with V7_PREP_OVERWRITE=1." >&2
    exit 1
fi

ARGS=(
    --chirps-glob "$CHIRPS_GLOB"
    --cpc-glob "$CPC_GLOB"
    --era5-glob "$ERA5_GLOB"
    --static "$STATIC"
    --out "$OUT"
    --chunk-days "$CHUNK_DAYS"
    --start "$START" --end "$END"
    --factor "$FACTOR"
    --coarse-res "$COARSE_RES"
    --smooth-base-iterations "$SMOOTH_ITERS"
)
[[ "$OVERWRITE" == "1" ]] && ARGS+=(--overwrite)

echo "Preparing the V7 archive"
echo "  support: ${COARSE_RES} deg coarse, factor ${FACTOR} to 0.05 deg fine"
echo "  period:  $START through $END   (no IMERG dependency -- observation only)"
echo "  output:  $OUT"
"$PYTHON_BIN" -u scripts/56_build_chirps_subgrid_targets.py "${ARGS[@]}"

"$PYTHON_BIN" - "$OUT" "$FACTOR" "$COARSE_RES" <<'PY'
import sys
import zarr
from bdhires.data import SUBGRID_SCHEMA

root = zarr.open_group(sys.argv[1], mode="r")
assert root.attrs.get("complete", False), "archive is not marked complete"
assert root.attrs.get("schema") == SUBGRID_SCHEMA
encoding = dict(root.attrs["subgrid_encoding"])
assert int(encoding["factor"]) == int(sys.argv[2]), "archive factor is not the V7 factor"
assert int(encoding["smooth_base_iterations"]) > 0, "the blockiness fix is absent"
assert float(encoding["amount_sqrt_std"]) > 0.0
assert float(encoding["intensity_log_std"]) > 0.0
fine = root["fine_mm"].shape[-2:]
coarse = root["coarse_mm"].shape[-2:]
assert fine[0] == coarse[0] * int(sys.argv[2]), "coarse/fine shapes disagree with the factor"
# CPC is coarser than a 0.1-degree support, so it must have been replicated, not
# selected -- if this says native the coarse grid is not the one V7 intends.
assert root.attrs.get("native_cpc") is False, (
    "CPC reports native on the V7 coarse grid; the support is not 0.1 degrees"
)
print("Validated V7 archive:", sys.argv[1])
print("  coarse", coarse, "fine", fine, "days", root["time"].shape[0])
PY
