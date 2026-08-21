#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v2-jun23-bgd
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bdhires-v2-jun23-bgd-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" logs

ROOT="${V2_CONFIRM_ROOT:-data/processed/v2_confirmatory_2021_2024}"
OUT="${V2_JUNE_BGD_OUT:-$ROOT/evaluation/june2023_bangladesh_ig010}"
BOUNDARY="${V2_JUNE_BGD_BOUNDARY:-$OUT/inputs/geoBoundaries-BGD-ADM0.geojson}"
METHOD="${V2_JUNE_BGD_METHOD:-v2_simul_s04_ig010}"
STORES=(
    "$ROOT/gridded/2021_may_sep.zarr"
    "$ROOT/gridded/2022_may_sep.zarr"
    "$ROOT/gridded/2023_may_sep.zarr"
    "$ROOT/gridded/2024_may_jun.zarr"
)

for required in "$BOUNDARY" "${STORES[@]}" scripts/81_evaluate_cpcv2_june_bangladesh.py; do
    [[ -f "$required" || -d "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done
if [[ -s "$OUT/evaluation.json" && "${V2_JUNE_BGD_FORCE:-0}" != "1" ]]; then
    echo "Evaluation already complete: $OUT/evaluation.json"
    echo "Set V2_JUNE_BGD_FORCE=1 to overwrite the report."
    exit 0
fi

echo "Evaluating saved CPCv2 product (no DA regeneration)"
echo "  method: $METHOD"
echo "  target: June 2023"
echo "  climatology candidates: complete Junes in 2021-2024 stores"
echo "  mask: Bangladesh ADM0 ($BOUNDARY)"
echo "  output: $OUT"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/81_evaluate_cpcv2_june_bangladesh.py \
    --zarr "${STORES[@]}" \
    --boundary-geojson "$BOUNDARY" \
    --method "$METHOD" \
    --year 2023 --month 6 --factor 8 \
    --out-dir "$OUT"

echo "Evaluation complete: $OUT/evaluation.json"
echo "Maps and plots: $OUT/{01..09}_*.png"
