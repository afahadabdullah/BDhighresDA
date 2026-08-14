#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v2-grid-eval
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bdhires-v2-grid-eval-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" logs

ROOT="${V2_CONFIRM_ROOT:-data/processed/v2_confirmatory_2021_2024}"
OUT="${V2_EVAL_OUT:?set V2_EVAL_OUT}"
IFS=: read -r -a STORES <<< "${V2_EVAL_ZARRS:?set colon-separated V2_EVAL_ZARRS}"
if (( ${#STORES[@]} < 1 )); then
    echo "ERROR: no Zarr stores supplied" >&2
    exit 1
fi
for store in "${STORES[@]}"; do
    [[ -d "$store" ]] || { echo "ERROR: missing completed store $store"; exit 1; }
done
if [[ -s "$OUT/evaluation.json" && "${V2_EVAL_FORCE:-0}" != "1" ]]; then
    echo "Evaluation already complete; reusing $OUT/evaluation.json"
    echo "Set V2_EVAL_FORCE=1 if CV folds finished after this report was written."
    exit 0
fi

echo "Evaluating ${#STORES[@]} completed seasonal Zarr store(s)"
printf '  %s\n' "${STORES[@]}"
echo "  output: $OUT"
echo "  CHIRPS/IMERG/CPC are agreement references, not asserted truth"

"$PYTHON_BIN" -u scripts/55_evaluate_v2_gridded_archive.py \
    --zarr "${STORES[@]}" \
    --cv-root "${V2_EVAL_CV_ROOT:-$ROOT}" \
    --factor 8 \
    --texture-members "${V2_EVAL_TEXTURE_MEMBERS:-5}" \
    --out-dir "$OUT"

echo "Evaluation complete: $OUT/evaluation.json"
echo "Figures: $OUT/fig{01..11}_*.png"
