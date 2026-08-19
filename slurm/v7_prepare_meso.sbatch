#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v7-prepare-meso
#SBATCH --partition=grace-cpuonly
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

if [[ -d "$OUT" && "$OVERWRITE" != "1" ]]; then
    echo "Reusing existing coarsened archive: $OUT"
else
    ARGS=(--in "$SOURCE" --out "$OUT" --factor "$FACTOR")
    [[ "$OVERWRITE" == "1" ]] && ARGS+=(--overwrite)
    "$PYTHON_BIN" -u scripts/71_coarsen_pack_archive.py "${ARGS[@]}"
fi

# Statistics belong to the archive they were measured on.  Reusing the
# 0.05-degree file here would normalise the model with the wrong moments.
"$PYTHON_BIN" -u scripts/06_compute_stats.py --zarr "$OUT" --out "$STATS"

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
