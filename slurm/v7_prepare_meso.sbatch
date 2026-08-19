#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v7-prepare-meso
# grace-cpuonly is capacity-limited, so this lists both partitions and lets
# Slurm take whichever frees first.  This job requests no --gres, so landing
# on a GPU node never reserves a GPU -- it uses that node's Grace cores only.
#SBATCH --partition=grace,grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

# Stage A inputs: coarsen the v2 packed archive to 0.1 degrees, then compute its
# statistics.  Nothing is re-packed and nothing is re-interpolated -- the CPCv2
# regridding stands, and this only reduces it by an exact area-weighted factor.

set -euo pipefail
module purge 2>/dev/null || true
export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: ARM Python not found: $PYTHON_BIN" >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the repository root}"
export PYTHONPATH="$PWD/src"
mkdir -p logs

SOURCE="${V7_V2_ARCHIVE:-data/processed/bd_wide_cpc.zarr}"
OUT="${V7_MESO_ARCHIVE:-data/processed/bd_wide_cpc_0p1.zarr}"
STATS="${V7_MESO_STATS:-data/processed/stats_v7_meso.json}"
FACTOR="${V7_COARSEN_FACTOR:-2}"
OVERWRITE="${V7_COARSEN_OVERWRITE:-0}"

[[ -d "$SOURCE" ]] || {
    echo "ERROR: missing the CPCv2 packed archive: $SOURCE" >&2
    echo "Build it with slurm/pack_training_data.sbatch first." >&2
    exit 1
}

# Reuse only a COMPLETE archive.  A directory left behind by a failed run looks
# exactly like a finished one to an existence test, and reusing it would train on
# a half-written file -- so completeness is checked against the source rather than
# against the filesystem.
REUSABLE=0
if [[ -d "$OUT" && "$OVERWRITE" != "1" ]]; then
    REUSABLE="$("$PYTHON_BIN" - "$OUT" "$SOURCE" "$FACTOR" <<'PY'
import sys
import zarr

try:
    out = zarr.open_group(sys.argv[1], mode="r")
    src = zarr.open_group(sys.argv[2], mode="r")
    grid = dict(out.attrs["grid"])
    ok = (
        int(out.attrs.get("coarsen_factor", 0)) == int(sys.argv[3])
        and out.attrs.get("coarsened_from") == sys.argv[2]
        and list(out.attrs["cond_channels"]) == list(src.attrs["cond_channels"])
        and out["time"].shape[0] == src["time"].shape[0]
        and out["target"].shape[0] == src["target"].shape[0]
        and tuple(out["target"].shape[-2:]) == (grid["nlat"], grid["nlon"])
        and out["cond"].shape[0] == src["cond"].shape[0]
    )
    print("1" if ok else "0")
except Exception:
    print("0")
PY
)"
fi

if [[ "$REUSABLE" == "1" ]]; then
    echo "Reusing complete coarsened archive: $OUT"
else
    if [[ -d "$OUT" ]]; then
        echo "Existing $OUT is incomplete or does not match its source; rebuilding"
    fi
    "$PYTHON_BIN" -u scripts/71_v7_coarsen_pack_archive.py \
        --in "$SOURCE" --out "$OUT" --factor "$FACTOR" --overwrite
    # Statistics must never outlive the archive they were measured on.
    rm -f "$STATS"
fi

if [[ -s "$STATS" ]]; then
    echo "Reusing statistics: $STATS"
else
    "$PYTHON_BIN" -u scripts/06_compute_stats.py --zarr "$OUT" --out "$STATS"
fi

"$PYTHON_BIN" - "$OUT" "$STATS" "$FACTOR" <<'PY'
import json
import sys
import zarr

root = zarr.open_group(sys.argv[1], mode="r")
grid = dict(root.attrs["grid"])
assert int(root.attrs["coarsen_factor"]) == int(sys.argv[3])
assert abs(grid["res"] - 0.05 * int(sys.argv[3])) < 1e-9, grid["res"]
stats = json.load(open(sys.argv[2]))
print("Validated V7 stage-A inputs")
print("  grid  :", grid["nlat"], "x", grid["nlon"], "@", grid["res"], "deg")
print("  source:", root.attrs["coarsened_from"])
print("  stats :", sys.argv[2], "keys", sorted(stats)[:4], "...")
PY
