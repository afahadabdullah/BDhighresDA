#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v3-prepare
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || {
    echo "ERROR: ARM Python not found: $PYTHON_BIN" >&2
    exit 1
}

cd "${SLURM_SUBMIT_DIR:?submit this job with sbatch from the repository root}"
export PYTHONPATH="$PWD/src"
mkdir -p logs

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: V3 preparation must run on an aarch64 Grace node" >&2
    exit 1
}
[[ -f scripts/56_build_chirps_subgrid_targets.py ]] || {
    echo "ERROR: submit from the BDhighresDA repository root" >&2
    exit 1
}

CHIRPS_GLOB="${V3_CHIRPS_GLOB:-data/raw/chirps_v3sg/chirps_wide_cpc_*.nc}"
CPC_GLOB="${V3_CPC_GLOB:-data/raw/cpc/precip.*.nc}"
ERA5_GLOB="${V3_ERA5_GLOB:-data/raw/era5/era5_daily_*.nc}"
STATIC="${V3_STATIC:-data/static/static_wide_cpc.nc}"
DEM="${V3_DEM:-data/raw/dem/copernicus_glo90_wide_cpc.nc}"
STATIC_CHIRPS="${V3_STATIC_CHIRPS:-}"
OUT="${V3_PREP_OUT:-data/processed/cpc_v3_subgrid/wide_cpc.zarr}"
CHUNK_DAYS="${V3_PREP_CHUNK_DAYS:-32}"
OVERWRITE="${V3_PREP_OVERWRITE:-0}"

if [[ -e "$OUT" ]]; then
    COMPLETE="$($PYTHON_BIN - "$OUT" <<'PY'
import sys
import zarr

try:
    root = zarr.open_group(sys.argv[1], mode="r")
    valid = (
        root.attrs.get("complete", False)
        and root.attrs.get("schema") == "cpc_v3_subgrid_v1"
        and root["fine_mm"].shape[-2:] == (260, 260)
        and root["coarse_mm"].shape[-2:] == (26, 26)
    )
    print("1" if valid else "0")
except Exception:
    print("0")
PY
)"
    if [[ "$COMPLETE" == "1" && "$OVERWRITE" != "1" ]]; then
        echo "V3 target archive already complete; reusing $OUT"
        exit 0
    fi
    if [[ "$OVERWRITE" != "1" ]]; then
        echo "ERROR: $OUT exists but is not a completed archive" >&2
        echo "Inspect it, or resubmit with V3_PREP_OVERWRITE=1 to replace it." >&2
        exit 1
    fi
fi

if [[ ! -f "$STATIC" ]]; then
    if [[ -z "$STATIC_CHIRPS" ]]; then
        mapfile -t CHIRPS_CANDIDATES < <(compgen -G "$CHIRPS_GLOB" | sort || true)
        if (( ${#CHIRPS_CANDIDATES[@]} == 0 )); then
            echo "ERROR: cannot build $STATIC because $CHIRPS_GLOB matched no files" >&2
            echo "Prepare the aligned wide_cpc CHIRPS inputs, then rerun the pipeline." >&2
            exit 1
        fi
        STATIC_CHIRPS="${CHIRPS_CANDIDATES[0]}"
    fi
    [[ -f "$STATIC_CHIRPS" ]] || {
        echo "ERROR: static-reference CHIRPS file is missing: $STATIC_CHIRPS" >&2
        exit 1
    }
    [[ -f "$DEM" ]] || {
        echo "ERROR: aligned DEM is missing: $DEM" >&2
        echo "Create it with the V3-SG --grid wide_cpc DEM command in the runbook." >&2
        exit 1
    }
    echo "Building missing V3 static predictors"
    echo "  CHIRPS reference: $STATIC_CHIRPS"
    echo "  DEM:              $DEM"
    echo "  output:           $STATIC"
    "$PYTHON_BIN" -u scripts/03_build_static.py \
        --grid wide_cpc --dem "$DEM" --chirps "$STATIC_CHIRPS" --out "$STATIC"
else
    echo "Reusing V3 static predictors: $STATIC"
fi

ARGS=(
    --chirps-glob "$CHIRPS_GLOB"
    --cpc-glob "$CPC_GLOB"
    --era5-glob "$ERA5_GLOB"
    --static "$STATIC"
    --out "$OUT"
    --chunk-days "$CHUNK_DAYS"
)
if [[ "$OVERWRITE" == "1" ]]; then
    ARGS+=(--overwrite)
fi

echo "Preparing CPC-aligned V3-SG archive"
echo "  CHIRPS: $CHIRPS_GLOB"
echo "  CPC:    $CPC_GLOB"
echo "  ERA5:   $ERA5_GLOB"
echo "  static: $STATIC"
echo "  output: $OUT"

"$PYTHON_BIN" -u scripts/56_build_chirps_subgrid_targets.py "${ARGS[@]}"

"$PYTHON_BIN" - "$OUT" <<'PY'
import sys
import zarr

root = zarr.open_group(sys.argv[1], mode="r")
assert root.attrs.get("complete", False), "archive is not marked complete"
assert root.attrs.get("schema") == "cpc_v3_subgrid_v1"
encoding = dict(root.attrs["subgrid_encoding"])
assert int(encoding["factor"]) == 10
assert root["fine_mm"].shape[-2:] == (260, 260)
assert root["coarse_mm"].shape[-2:] == (26, 26)
print("Validated completed V3 archive:", sys.argv[1])
print("Days:", root["time"].shape[0])
PY
