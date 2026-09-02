#!/usr/bin/env bash
#SBATCH --job-name=bdhires-bwdb-bgd-eval
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: ARM Python not found: $PYTHON_BIN" >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: this job must run on an aarch64 Grace/GH200 node" >&2
    exit 1
}
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bdhires-bwdb-bgd-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" logs

ROOT="${V2_BWDB_ROOT:-data/processed/v2_bmd_bwdb_huber3_2021_2024}"
IMERG_NATIVE_ROOT="${V2_IMERG_NATIVE_ROOT:-data/processed/v2_confirmatory_2021_2024/imerg_native}"
MONTHS_TEXT="${BRISHTI_MONTHS:-6}"
read -r -a MONTHS <<< "$MONTHS_TEXT"
PERIOD_TAG="${BRISHTI_PERIOD_TAG:-jun}"
YEAR="${BRISHTI_YEAR:-2023}"
OUT="${V2_BWDB_BGD_OUT:-$ROOT/evaluation/brishti05_${PERIOD_TAG}${YEAR}_native_refs}"
BOUNDARY="${V2_BWDB_BGD_BOUNDARY:-$ROOT/evaluation/brishti05_inputs/geoBoundaries-BGD-ADM0.geojson}"
METHOD="${V2_BWDB_METHOD:-v2_simul_s04_huber3}"
CPC_DIR="${V2_BWDB_CPC_DIR:-data/raw/cpc}"
STORES=(
    "$ROOT/gridded/2021_may_sep.zarr"
    "$ROOT/gridded/2022_may_sep.zarr"
    "$ROOT/gridded/2023_may_sep.zarr"
    "$ROOT/gridded/2024_may_jun.zarr"
)
NATIVE_IMERG=(
    "$IMERG_NATIVE_ROOT/2021_may_sep.nc"
    "$IMERG_NATIVE_ROOT/2022_may_sep.nc"
    "$IMERG_NATIVE_ROOT/2023_may_sep.nc"
    "$IMERG_NATIVE_ROOT/2024_may_jun.nc"
)

for required in "$BOUNDARY" "${STORES[@]}" "${NATIVE_IMERG[@]}" \
                "$CPC_DIR/precip.2021.nc" "$CPC_DIR/precip.2022.nc" \
                "$CPC_DIR/precip.2023.nc" "$CPC_DIR/precip.2024.nc" \
                scripts/85_evaluate_bmd_bwdb_bangladesh.py; do
    [[ -f "$required" || -d "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done
if [[ -s "$OUT/evaluation.json" && "${V2_BWDB_BGD_FORCE:-0}" != "1" ]]; then
    echo "Evaluation already complete: $OUT/evaluation.json"
    echo "Set V2_BWDB_BGD_FORCE=1 to overwrite the report."
    exit 0
fi

echo "Evaluating BMD+BWDB Huber3 analysis product over Bangladesh ADM0"
echo "  method: $METHOD"
echo "  requested year: $YEAR"
echo "  requested months: ${MONTHS[*]}"
echo "  climatology candidates: complete matching periods in saved stores"
echo "  comparison products: native 0.1-degree IMERG; original 0.5-degree CPC"
echo "  mask: Bangladesh ADM0 ($BOUNDARY)"
echo "  output: $OUT"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/85_evaluate_bmd_bwdb_bangladesh.py \
    --zarr "${STORES[@]}" \
    --boundary-geojson "$BOUNDARY" \
    --native-imerg "${NATIVE_IMERG[@]}" \
    --cpc-dir "$CPC_DIR" \
    --method "$METHOD" \
    --year "$YEAR" --months "${MONTHS[@]}" --factor 8 \
    --out-dir "$OUT"

echo "Evaluation complete: $OUT/evaluation.json"
echo "Maps and plots: $OUT/{01..10}_*.png"
