#!/usr/bin/env bash
#SBATCH --job-name=graphflow-g0-da-may22
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-4%2
#SBATCH --output=logs/%x-%A_%a.out

# One GraphFlow G0 fold using the exact frozen CPCv2 DA contract. The existing
# CPCv2 May 1--10 folds are read-only references and are never rerun.

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN"; exit 1; }
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD/src"
mkdir -p logs

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: this job must run on an aarch64 Grace/GH200 node" >&2
    exit 1
}

"$PYTHON_BIN" -u scripts/51_check_sqrt_da_gradient.py

FOLD="${SLURM_ARRAY_TASK_ID:?run as the configured five-fold array}"
START="2022-05-01"
END="2022-05-10"
ROOT="${GRAPHFLOW_DA_ROOT:-runs/graphflow_g0_multimesh/da_may2022}"
CPC_ROOT="${GRAPHFLOW_CPC_ROOT:-data/processed/v2_simultaneous_refinement/ing2022_s04}"
IMERG="${GRAPHFLOW_IMERG:-data/processed/imerg_prepared_ing2022/imerg_0p4deg_20220501_20220510.nc}"
GRAPH_CKPT="${GRAPHFLOW_CKPT:-runs/prior_h100_cpc_graphflow_g0/best.pt}"
CPC_CKPT="${BMD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
STATIONS="$CPC_ROOT/fold${FOLD}_bmd.csv"
PREFIX="$ROOT/graphflow_g0_multimesh/fold${FOLD}"

for required in "$GRAPH_CKPT" "$CPC_CKPT" "$IMERG" "$STATIONS" \
                "$CPC_ROOT/fold${FOLD}.npz" "$CPC_ROOT/fold${FOLD}.json"; do
    [[ -s "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done

if [[ -s "${PREFIX}.npz" && -s "${PREFIX}.json" ]]; then
    echo "GraphFlow fold $FOLD already complete; reusing ${PREFIX}.{npz,json}"
    exit 0
fi
if [[ -e "${PREFIX}.npz" || -e "${PREFIX}.json" ]]; then
    echo "ERROR: fold $FOLD has only one output; move the partial file aside before retrying" >&2
    exit 1
fi

echo "GraphFlow G0 frozen DA, fold $((FOLD + 1))/5"
echo "  period: $START through $END; members: 30"
echo "  GraphFlow checkpoint: $GRAPH_CKPT"
echo "  existing CPCv2 reference: $CPC_ROOT/fold${FOLD}.{npz,json}"
echo "  output: $PREFIX"

"$PYTHON_BIN" -u scripts/graphflow_experiment.py da \
    --model graphflow --fold "$FOLD" \
    --unet-ckpt "$CPC_CKPT" --graphflow-ckpt "$GRAPH_CKPT" \
    --stations "$STATIONS" --imerg "$IMERG" \
    --start "$START" --end "$END" --out "$ROOT"

echo "GraphFlow fold $FOLD complete: ${PREFIX}.{npz,json}"
