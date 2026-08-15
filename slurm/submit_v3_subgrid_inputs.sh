#!/usr/bin/env bash
set -euo pipefail

# Build V3-specific aligned CHIRPS, then the aligned DEM/static predictors.
# Existing valid outputs are reused by their underlying preparation scripts.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

for required in \
    scripts/01_download_chirps.py \
    scripts/03_download_dem.py \
    scripts/03_build_static.py \
    slurm/v3_subgrid_chirps.sbatch \
    slurm/download_dem_static.sbatch; do
    [[ -f "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done

export V3_CHIRPS_START="${V3_CHIRPS_START:-1981}"
export V3_CHIRPS_END="${V3_CHIRPS_END:-2024}"
export V3_CHIRPS_MAX_CONCURRENT="${V3_CHIRPS_MAX_CONCURRENT:-4}"
export V3_CHIRPS_OUT="${V3_CHIRPS_OUT:-data/raw/chirps_v3sg}"
if (( V3_CHIRPS_START > V3_CHIRPS_END )); then
    echo "ERROR: V3_CHIRPS_START must not exceed V3_CHIRPS_END" >&2
    exit 1
fi
if (( V3_CHIRPS_MAX_CONCURRENT < 1 )); then
    echo "ERROR: V3_CHIRPS_MAX_CONCURRENT must be positive" >&2
    exit 1
fi

# These project sources can be reused. Script 56 later checks their exact
# coordinates and coverage before writing target data.
for label_and_pattern in \
    "CPC|data/raw/cpc/precip.*.nc" \
    "ERA5|data/raw/era5/era5_daily_*.nc"; do
    label="${label_and_pattern%%|*}"
    pattern="${label_and_pattern#*|}"
    if ! compgen -G "$pattern" >/dev/null; then
        echo "ERROR: existing $label pattern matched no files: $pattern" >&2
        echo "Prepare that source before submitting V3-specific inputs." >&2
        exit 1
    fi
done

MISSING_CPC=()
MISSING_ERA5=()
for ((year=V3_CHIRPS_START; year<=V3_CHIRPS_END; year++)); do
    [[ -f "data/raw/cpc/precip.${year}.nc" ]] || MISSING_CPC+=("$year")
    [[ -f "data/raw/era5/era5_daily_${year}.nc" ]] || MISSING_ERA5+=("$year")
done
if (( ${#MISSING_CPC[@]} )); then
    echo "ERROR: CPC is missing requested years: ${MISSING_CPC[*]}" >&2
    exit 1
fi
if (( ${#MISSING_ERA5[@]} )); then
    echo "ERROR: ERA5 is missing requested years: ${MISSING_ERA5[*]}" >&2
    exit 1
fi

TASKS=$((V3_CHIRPS_END - V3_CHIRPS_START + 1))
LAST_TASK=$((TASKS - 1))
STATIC_YEAR=2010
if (( STATIC_YEAR < V3_CHIRPS_START || STATIC_YEAR > V3_CHIRPS_END )); then
    STATIC_YEAR="$V3_CHIRPS_START"
fi
export STATIC_GRID="wide_cpc"
export CHIRPS_REFERENCE="$V3_CHIRPS_OUT/chirps_wide_cpc_${STATIC_YEAR}.nc"
export DEM_OUT="data/raw/dem/copernicus_glo90_wide_cpc.nc"
export DEM_TILE_DIR="${DEM_TILE_DIR:-data/raw/dem/copernicus_glo90_tiles}"
export STATIC_OUT="data/static/static_wide_cpc.nc"

echo "Submitting V3-specific aligned inputs"
echo "  CHIRPS years: $V3_CHIRPS_START..$V3_CHIRPS_END"
echo "  array:        0-$LAST_TASK%$V3_CHIRPS_MAX_CONCURRENT"
echo "  DEM:          $DEM_OUT"
echo "  static:       $STATIC_OUT"
echo "  ERA5 reused:  data/raw/era5/era5_daily_*.nc"

chirps_result="$(sbatch --parsable \
    --array="0-${LAST_TASK}%${V3_CHIRPS_MAX_CONCURRENT}" \
    --export=ALL "$@" slurm/v3_subgrid_chirps.sbatch)"
chirps_job="${chirps_result%%;*}"

static_result="$(sbatch --parsable --job-name=bdhires-v3-static \
    --dependency="afterok:${chirps_job}" --export=ALL "$@" \
    slurm/download_dem_static.sbatch)"
static_job="${static_result%%;*}"

echo "submitted CHIRPS array: $chirps_job"
echo "submitted DEM/static:   $static_job (afterok:$chirps_job)"
echo "monitor: squeue -u $USER"
echo "logs:    logs/bdhires-v3-chirps-${chirps_job}_*.out"
echo "         logs/bdhires-v3-static-${static_job}.out"
echo "after both finish: bash slurm/submit_v3_subgrid_pipeline.sh"
